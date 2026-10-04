#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RESS V1-I.6-TURBO.1 worker
Shared exogenous paths + batched gamma evaluation + outer-path checkpoint/resume.

Scientific design is unchanged:
  BTC M=512, ETH M=256, L=2048, Gamma={0,.1,...,1}, alpha=.95, eps=delta=.08.
The speed-up is computational only: for a fixed (asset,m,l), one exogenous market
path is generated and all gamma policies are evaluated on it.

TRAIN checkpoint unit = one completed outer path m, storing an (L,11) cost block.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")

import argparse, importlib.util, json, math, time, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

HASH="0791be105f4afd18170c7e5b12a74078afc684815414f713ecb80f46a99bcf50"
AS=("BTCUSDT","ETHUSDT")
OB=("expected_cost","nominal_cvar","global_kl","localized_kl")
GG=np.round(np.linspace(0,1,11),1)
EPSGRID=[0.,.02,.04,.08,.12,.16]
SEEDS=[101,202,303,404,505,606,707,808,909,1001,1111,1212,1313,1414,1515]

def lm(p,n):
    s=importlib.util.spec_from_file_location(n,p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def js(p): return json.loads(Path(p).read_text())
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2))
def split_windows(d3,raw,split,H,grid):
    z=raw.copy()
    z["split"]=np.where(z["split"].astype(str).str.lower().eq(split),"train","excluded")
    return d3.build_train_windows(z,H,grid)
def cvar(x,a=.95):
    x=np.asarray(x,float).ravel()
    q=float(np.quantile(x,a,method="lower"))
    return float(q+np.maximum(x-q,0).mean()/(1-a)),q
def klweights(G,e,d3): return d3._kl_sup_weights(np.asarray(G,float),e)

# SciPy-free bounded golden-section minimizer: avoids a second scientific runtime.
def _golden(f,lo,hi,tol=None,maxiter=140):
    lo=float(lo); hi=float(hi)
    if not np.isfinite(lo+hi): raise RuntimeError("nonfinite optimization bounds")
    if hi<=lo: return lo
    if tol is None: tol=max(1e-13,(hi-lo)*1e-10)
    gr=(math.sqrt(5.0)-1.0)/2.0
    c=hi-gr*(hi-lo); d=lo+gr*(hi-lo); fc=f(c); fd=f(d)
    for _ in range(maxiter):
        if hi-lo<=tol: break
        if fc<=fd:
            hi,d,fd=d,c,fc; c=hi-gr*(hi-lo); fc=f(c)
        else:
            lo,c,fc=c,d,fd; d=lo+gr*(hi-lo); fd=f(d)
    pts=[lo,hi,c,d,(lo+hi)/2]
    vals=[f(x) for x in pts]
    return float(pts[int(np.argmin(vals))])

def loc_fast(C,a,e,d3):
    C=np.asarray(C,float); lo=float(C.min()); hi=float(C.max())
    def f(eta):
        G=np.maximum(C-eta,0).mean(1); sup,_,_,_=klweights(G,e,d3)
        return float(eta+sup/(1-a))
    eta=_golden(f,lo,hi)
    G=np.maximum(C-eta,0).mean(1); sup,q,lam,kl=klweights(G,e,d3)
    return dict(rho=float(eta+sup/(1-a)),eta=eta,lambda_=lam,kl=kl,
                neff=float(1/(q@q)),qmax=float(q.max()))
def glob_fast(x,a,e,d3):
    x=np.asarray(x,float).ravel(); lo=float(x.min()); hi=float(x.max())
    def f(eta):
        G=np.maximum(x-eta,0); sup,_,_,_=klweights(G,e,d3)
        return float(eta+sup/(1-a))
    eta=_golden(f,lo,hi)
    G=np.maximum(x-eta,0); sup,q,lam,kl=klweights(G,e,d3)
    return dict(rho=float(eta+sup/(1-a)),eta=eta,lambda_=lam,kl=kl,
                neff=float(1/(q@q)),qmax=float(q.max()))

def setup(a):
    d3=lm(a.d3_script,"d3_i6turbo")
    h=lm(a.h33_script,"h33_i6turbo")
    d3.bind_h33(h,type("D",(),{}))
    raw,_=h.load_panel(a.panel); raw=h.build_raw(raw,5,10)
    import torch,torch.nn as nn,torch.nn.functional as F
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    dev=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    model,xcols,ln,sc,cf,cfg=h.load_frozen_m2(a.vep_results,dev,torch,nn,F)
    liq=h.load_json(Path(a.g55_results)/"models"/"v1g55_liquidity_varx.json")
    spr=h.load_json(Path(a.g55_results)/"models"/"v1g55_spread_arx.json")
    sres=h.load_npz_resid(Path(a.g55_results)/"models"/"v1g55_spread_residuals.npz")
    book=h.load_empirical_book(a.h22_results,10)
    with torch.inference_mode():
        nu,R=model.noise(); noise=(float(nu.cpu()),R.cpu().numpy())
    return d3,h,raw,torch,dev,model,ln,sc,cf,liq,spr,sres,book,noise

def theta_bank(d3,df,starts,asset,M,seed,ctx):
    h,torch,dev,model,ln,sc,*_=ctx
    ids=[i for i,x in enumerate(starts) if x[1]==asset]
    if len(ids)<M: raise RuntimeError(f"{asset}: only {len(ids)} starts for M={M}")
    rng=np.random.default_rng(np.random.SeedSequence([seed,610001,1 if asset=="BTCUSDT" else 2]))
    ch=rng.choice(ids,size=M,replace=False); B=[]
    for j in ch:
        st,_,date=starts[int(j)]
        th,_,_=d3.sample_reference_theta_path(df,st,120,model,ln,sc,rng,torch,dev)
        B.append((st,np.asarray(th,float),date))
    return B

def reference_episode(d3,r0,asset,gamma,theta,seed,ctx):
    h,torch,dev,model,ln,sc,cf,liq,spr,sres,book,noise=ctx
    pol=d3.GammaPolicy(float(gamma),torch,dev)
    z=d3.simulate_policy_episode_fixed_theta(
        r0,asset,120,2.,pol,.1,10,model,ln,sc,cf,liq,spr,sres,book,
        np.random.default_rng(int(seed)),torch,dev,12,noise,960,theta)
    if z["numerical_failure"] or z["terminal_settlement_exhausted"]:
        raise RuntimeError("reference episode failure")
    return z

def turbo_episode(d3,r0,sym,theta_path,seed,ctx,gammas=GG,H=120,
                  inv_frac=2.,max_participation=.1,K=10,rv_window=12,max_terminal_steps=960):
    """
    Exact shared-exogenous-path evaluator for the current D.3 environment.
    The exogenous LOB transition is action-independent. All gamma policies therefore
    see the same sampled books and innovations under CRN; only inventory/cost differs.
    """
    h,torch,device,model,ln,sc,center_fits,liq,spread_spec,spread_resid,book,noise_cache=ctx
    E=float(d3.EPS); XI=d3.XI; XFULL=d3.XFULL
    rng=np.random.default_rng(int(seed))
    gammas=np.asarray(gammas,float)
    # Preserve exactly the reference GammaPolicy -> torch -> .item() numerical path.
    policies=[d3.GammaPolicy(float(g),torch,device) for g in gammas]

    mid=float(r0.mid); total=float(r0.total_depth); ratio=float(r0.log_depth_ratio)
    spread_bps=float(r0.spread_bps); imb=float(r0.imbalance)
    rv=max(float(r0.rv_backward),E)
    last=np.array([float(r0[c]) for c in XI])
    db,_,_=d3.reconstruct_depth(total,ratio)
    q0=max(inv_frac*db,E); q=np.full(len(gammas),q0,float)
    mid0=max(mid,E); depth0=max(total,E)
    sf=np.zeros(len(gammas),float); regular_sf=np.zeros(len(gammas),float)
    filled=np.zeros(len(gammas),float)
    rv_window=max(int(rv_window),1); rets=[rv/math.sqrt(rv_window)]*rv_window
    numerical_fail=False; theta_n=0

    for t in range(H):
        if not all(np.isfinite(v) for v in [mid,total,ratio,spread_bps,rv]) or min(mid,total,spread_bps,rv)<=0:
            numerical_fail=True; break
        db,da,imb=d3.reconstruct_depth(total,ratio)
        spread=mid*spread_bps/1e4
        shares,mults,exact_cell,_=d3.sample_book_shape(sym,spread_bps,total,imb,book,rng,K)

        # Reproduce the scalar reference policy evaluation exactly. Although the
        # policy is constant in gamma, the reference obtains frac through torch
        # from a float32 feature tensor; using Python gamma directly changes a few
        # last bits (visible especially at gamma=0.9).
        feat=np.array([
            1.0,
            (H-t)/max(H,1),
            math.log(max(mid,E)/mid0),
            math.log(max(spread_bps,E)),
            math.log(max(total,E)/depth0),
            imb,
            math.log(max(rv,E)),
            last[0],last[1],last[2],
            1.0 if sym=="ETHUSDT" else 0.0
        ],dtype=np.float32)
        for j,pol in enumerate(policies):
            feat[0]=q[j]/max(q0,E)
            with torch.no_grad():
                xt=torch.as_tensor(feat,device=device).unsqueeze(0)
                frac=float(pol(xt).item())
            cap=min(float(q[j]),max(float(max_participation),0.0)*max(float(db),0.0))
            a=cap*min(max(frac,0.0),1.0)
            fill,vwap,cost,_=d3.walk_bid(mid,spread,db,a,shares,mults)
            q[j]-=fill; filled[j]+=fill; sf[j]+=cost; regular_sf[j]+=cost

        xraw=np.array([last[0],last[1],last[2],math.log(spread_bps),imb,math.log(rv)])
        xstd=np.array([(xraw[j]-float(sc[XFULL[j]]["mean"]))/
                       max(float(sc[XFULL[j]]["sd"]),E) for j in range(len(XFULL))])
        if theta_n>=len(theta_path): raise RuntimeError("fixed theta path exhausted")
        theta=float(theta_path[theta_n])
        if (not np.isfinite(theta)) or theta<=0: raise RuntimeError("invalid fixed theta")
        e=d3.sample_eps_given_theta(theta,rng,noise_cache); theta_n+=1
        cstd=np.asarray(d3.center_std(sym,xstd,center_fits),float)
        dmid=d3.inv_std("dlog_mid",cstd[0]+e[0],sc)
        y=np.array([math.log(total),ratio])
        zliq=np.array([last[0],math.log(spread_bps),imb,math.log(rv)])
        lc=d3.liq_center(liq[sym],y,zliq); rs=np.asarray(liq[sym]["resid_sd"],float)
        yn=lc+rs*np.asarray(e[1:3],float)
        total_n=math.exp(float(yn[0])); ratio_n=float(yn[1]); mid_n=math.exp(math.log(mid)+dmid)
        ddepth=math.log(total_n)-math.log(total); dratio=ratio_n-ratio
        sf_feat={"imbalance":imb,"log_rv_now":math.log(rv),"dlog_mid":dmid,
                 "dlog_total_depth":ddepth,"d_log_depth_ratio":dratio}
        spread_n=d3.spread_next(spread_spec[sym],spread_bps,sf_feat,rng,spread_resid[sym])
        rets.append(dmid); rets=rets[-rv_window:]
        rv_n=max(float(np.sqrt(np.sum(np.square(rets)))),E)
        mid,total,ratio,spread_bps,rv=mid_n,total_n,ratio_n,spread_n,rv_n
        last=np.array([dmid,ddepth,dratio])

    if theta_n!=H: raise RuntimeError(f"theta consumption {theta_n}!={H}")
    q_pre=np.maximum(q,0.).copy()
    terminal_steps=np.zeros(len(gammas),int)
    terminal_sf=np.zeros(len(gammas),float)
    settlement_numerical_failure=False

    # Generate one common nominal post-H continuation until every gamma is settled.
    shared_step=0
    while np.any(q>1e-12) and shared_step<max_terminal_steps and not numerical_fail:
        if not all(np.isfinite(v) for v in [mid,total,ratio,spread_bps,rv]) or min(mid,total,spread_bps,rv)<=0:
            settlement_numerical_failure=True; break
        db,da,imb=d3.reconstruct_depth(total,ratio)
        spread=mid*spread_bps/1e4
        shares,mults,exact_cell,_=d3.sample_book_shape(sym,spread_bps,total,imb,book,rng,K)

        active=np.flatnonzero(q>1e-12)
        for j in active:
            f,vwap,cst,residual=d3.forced_terminal_walk(mid,spread,db,float(q[j]),shares,mults)
            filled[j]+=f; terminal_sf[j]+=cst; sf[j]+=cst
            q[j]=residual; terminal_steps[j]+=1
            if q[j]<=1e-12: q[j]=0.0

        shared_step+=1
        if not np.any(q>1e-12): break

        xraw=np.array([last[0],last[1],last[2],math.log(spread_bps),imb,math.log(rv)])
        xstd=np.array([(xraw[j]-float(sc[XFULL[j]]["mean"]))/
                       max(float(sc[XFULL[j]]["sd"]),E) for j in range(len(XFULL))])
        with torch.inference_mode():
            xt_prior=torch.as_tensor(np.asarray(xstd,np.float32)[None,:],device=device)
            mt,st=model.prior.moments(xt_prior,ln)
            mpost=float(mt.detach().cpu()[0]); spost=float(st.detach().cpu()[0])
        theta=math.exp(rng.normal(mpost,spost))
        if (not np.isfinite(theta)) or theta<=0: raise RuntimeError("invalid nominal settlement theta")
        e=d3.sample_eps_given_theta(theta,rng,noise_cache)
        cstd=np.asarray(d3.center_std(sym,xstd,center_fits),float)
        dmid=d3.inv_std("dlog_mid",cstd[0]+e[0],sc)
        y=np.array([math.log(total),ratio])
        zliq=np.array([last[0],math.log(spread_bps),imb,math.log(rv)])
        lc=d3.liq_center(liq[sym],y,zliq); rs=np.asarray(liq[sym]["resid_sd"],float)
        yn=lc+rs*np.asarray(e[1:3],float)
        total_n=math.exp(float(yn[0])); ratio_n=float(yn[1]); mid_n=math.exp(math.log(mid)+dmid)
        ddepth=math.log(total_n)-math.log(total); dratio=ratio_n-ratio
        sf_feat={"imbalance":imb,"log_rv_now":math.log(rv),"dlog_mid":dmid,
                 "dlog_total_depth":ddepth,"d_log_depth_ratio":dratio}
        spread_n=d3.spread_next(spread_spec[sym],spread_bps,sf_feat,rng,spread_resid[sym])
        rets.append(dmid); rets=rets[-rv_window:]
        rv_n=max(float(np.sqrt(np.sum(np.square(rets)))),E)
        mid,total,ratio,spread_bps,rv=mid_n,total_n,ratio_n,spread_n,rv_n
        last=np.array([dmid,ddepth,dratio])

    exhausted=q>1e-12
    if numerical_fail or settlement_numerical_failure or np.any(exhausted):
        raise RuntimeError(f"turbo episode failure: numerical={numerical_fail or settlement_numerical_failure}, exhausted={np.flatnonzero(exhausted).tolist()}")
    arrival=float(r0.mid)
    costs=1e4*sf/(arrival*q0) if arrival>0 else np.full(len(gammas),np.nan)
    completion=filled/q0
    return {"shortfall_bps":np.asarray(costs,float),
            "completion_rate":np.asarray(completion,float),
            "terminal_steps":terminal_steps}

def equivalence_gate(d3,df,B,asset,ctx):
    """
    Mandatory exact/near-exact comparison against the original scalar simulator.
    Covers all 11 gammas, two outer atoms, two inner seeds = 44 scalar comparisons.
    """
    ac=1 if asset=="BTCUSDT" else 2
    max_abs=0.; max_rel=0.; n=0
    for m in range(min(2,len(B))):
        st,th,_=B[m]; r0=df.iloc[st]
        for l in range(2):
            seed=d3._seed(101,ac,m,l)
            fast=turbo_episode(d3,r0,asset,th,seed,ctx)["shortfall_bps"]
            ref=np.array([reference_episode(d3,r0,asset,g,th,seed,ctx)["shortfall_bps"] for g in GG])
            ae=np.abs(fast-ref); re=ae/np.maximum(np.abs(ref),1e-14)
            max_abs=max(max_abs,float(ae.max())); max_rel=max(max_rel,float(re.max())); n+=len(GG)
            if not np.allclose(fast,ref,rtol=1e-10,atol=1e-12):
                j=int(np.argmax(ae))
                raise RuntimeError(f"TURBO equivalence FAIL {asset} m={m} l={l} gamma={GG[j]:.1f}: fast={fast[j]:.17g} ref={ref[j]:.17g} abs={ae[j]:.3e}")
    print(f"TURBO EQUIVALENCE PASS | {asset} | n={n} | max_abs={max_abs:.3e} | max_rel={max_rel:.3e}",flush=True)

def _atomic_save_npz(path,**kw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp.npz")
    np.savez_compressed(tmp,**kw)
    os.replace(tmp,path)

def train(a,d3,raw,ctx,out):
    df,starts=split_windows(d3,raw,"train",120,5)
    rows=[]
    for asset in AS:
        M=512 if asset=="BTCUSDT" else 256; L=2048
        B=theta_bank(d3,df,starts,asset,M,101,ctx)
        equivalence_gate(d3,df,B,asset,ctx)
        ckdir=out/"train_outer"/asset; ckdir.mkdir(parents=True,exist_ok=True)
        ac=1 if asset=="BTCUSDT" else 2

        def one_outer(m):
            fp=ckdir/f"outer_{m:04d}.npz"
            if fp.exists():
                z=np.load(fp)
                C=z["costs"]
                if C.shape!=(L,len(GG)) or not np.all(np.isfinite(C)):
                    raise RuntimeError(f"bad checkpoint {fp}")
                return m,C,True
            st,th,_=B[m]; r0=df.iloc[st]
            C=np.empty((L,len(GG)),float)
            for l in range(L):
                seed=d3._seed(101,ac,m,l)
                C[l]=turbo_episode(d3,r0,asset,th,seed,ctx)["shortfall_bps"]
            _atomic_save_npz(fp,costs=C,gammas=GG,outer_id=np.array([m],int))
            return m,C,False

        done=0; reused=0
        blocks=[None]*M
        if a.jobs==1:
            iterator=(one_outer(m) for m in range(M))
            for m,C,reuse in iterator:
                blocks[m]=C; done+=1; reused+=int(reuse)
                if done==1 or done%4==0 or done==M:
                    print(f"TURBO TRAIN {asset} outer={done}/{M} | reused={reused} | each outer evaluates all 11 gamma",flush=True)
        else:
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                fut=[ex.submit(one_outer,m) for m in range(M)]
                for f in as_completed(fut):
                    m,C,reuse=f.result(); blocks[m]=C; done+=1; reused+=int(reuse)
                    if done==1 or done%4==0 or done==M:
                        print(f"TURBO TRAIN {asset} outer={done}/{M} | jobs={a.jobs} | reused={reused} | all-gamma",flush=True)

        cube=np.stack(blocks,axis=0)  # M x L x 11
        if cube.shape!=(M,L,len(GG)) or not np.all(np.isfinite(cube)):
            raise RuntimeError(f"{asset}: invalid completed cube {cube.shape}")
        # Save conventional per-gamma matrices for downstream compatibility.
        (out/"train").mkdir(parents=True,exist_ok=True)
        for j,gamma in enumerate(GG):
            C=cube[:,:,j]
            np.save(out/"train"/f"{asset}_g{gamma:.1f}.npy",C)
            ec=float(C.mean()); cv,_=cvar(C)
            loc=loc_fast(C,.95,.08,d3); glo=glob_fast(C,.95,.08,d3)
            if loc["rho"]+1e-12<cv: raise RuntimeError(f"{asset} g={gamma}: localized < nominal")
            if glo["rho"]+1e-12<loc["rho"]: raise RuntimeError(f"{asset} g={gamma}: global < localized")
            for o,v in [("expected_cost",ec),("nominal_cvar",cv),
                        ("global_kl",glo["rho"]),("localized_kl",loc["rho"])]:
                rows.append(dict(asset=asset,gamma=float(gamma),objective=o,value=float(v)))

    R=pd.DataFrame(rows); R.to_csv(out/"train_objectives.csv",index=False)
    sel=[]
    for (asset,o),g in R.groupby(["asset","objective"]):
        z=g.sort_values(["value","gamma"]).iloc[0]
        sel.append(dict(asset=asset,objective=o,gamma=float(z.gamma),train_value=float(z.value)))
    pd.DataFrame(sel).to_csv(out/"TRAIN_SELECTION.csv",index=False)

    sens=[]
    for z in sel:
        asset,o,gamma=z["asset"],z["objective"],float(z["gamma"])
        C=np.load(out/"train"/f"{asset}_g{gamma:.1f}.npy"); cv,_=cvar(C)
        for e in EPSGRID:
            loc=loc_fast(C,.95,e,d3) if e>0 else {"rho":cv}
            glo=glob_fast(C,.95,e,d3) if e>0 else {"rho":cv}
            sens.append(dict(asset=asset,selected_for=o,gamma=gamma,epsilon=e,
                             nominal_cvar=cv,localized_kl=loc["rho"],global_kl=glo["rho"]))
    pd.DataFrame(sens).to_csv(out/"TRAIN_RADIUS_SENSITIVITY.csv",index=False)
    dump(out/"TRAIN_LOCK.json",{
        "version":"V1-I.6-TURBO.1-TRAIN","locked":True,"selection":sel,
        "BTC_D3_outer_gate":"FAIL retained","shared_exogenous_paths":True,
        "batched_gamma_evaluation":True,"outer_checkpoint_resume":True,
        "no_validation":True,"no_test":True})
    print("TURBO TRAIN LOCKED",flush=True)

def validation(a,d3,raw,ctx,out):
    lock=js(out/"TRAIN_LOCK.json"); df,starts=split_windows(d3,raw,"validation",120,5); rows=[]
    for asset in AS:
        gs=sorted({float(x["gamma"]) for x in lock["selection"] if x["asset"]==asset})
        B=theta_bank(d3,df,starts,asset,64,202,ctx); ac=1 if asset=="BTCUSDT" else 2
        idx=[int(np.where(np.isclose(GG,g))[0][0]) for g in gs]
        C={g:np.empty((64,512),float) for g in gs}
        for m in range(64):
            st,th,_=B[m]; r0=df.iloc[st]
            for l in range(512):
                v=turbo_episode(d3,r0,asset,th,d3._seed(202,ac,m,l),ctx)["shortfall_bps"]
                for g,j in zip(gs,idx): C[g][m,l]=v[j]
            if m==0 or (m+1)%8==0: print(f"TURBO VAL {asset} {m+1}/64",flush=True)
        for gamma in gs:
            X=C[gamma]; ec=float(X.mean()); cv,_=cvar(X); loc=loc_fast(X,.95,.08,d3); glo=glob_fast(X,.95,.08,d3)
            vals={"expected_cost":ec,"nominal_cvar":cv,"global_kl":glo["rho"],"localized_kl":loc["rho"]}
            for o in OB:
                if any(x["asset"]==asset and x["objective"]==o and float(x["gamma"])==gamma for x in lock["selection"]):
                    rows.append(dict(asset=asset,objective=o,locked_gamma=gamma,validation_value=vals[o]))
    pd.DataFrame(rows).to_csv(out/"VALIDATION_CONFIRMATION.csv",index=False)
    dump(out/"VALIDATION_LOCK.json",{"version":"V1-I.6-TURBO.1-VALIDATION",
         "confirmed_without_retuning":True,"rows":rows,"no_test":True})
    print("TURBO VALIDATION LOCKED | NO RETUNING",flush=True)

def test(a,d3,raw,ctx,out):
    # TEST keeps CRN and deduplicates identical selected gamma. Reliability threshold
    # is frozen from TRAIN expected-cost policy VaR95, not estimated on TEST.
    lock=js(out/"TRAIN_LOCK.json"); df,starts=split_windows(d3,raw,"test",120,5); rows=[]
    thresholds={}
    for asset in AS:
        ecsel=next(x for x in lock["selection"] if x["asset"]==asset and x["objective"]=="expected_cost")
        X=np.load(out/"train"/f"{asset}_g{float(ecsel['gamma']):.1f}.npy")
        thresholds[asset]=float(np.quantile(X.ravel(),.95,method="lower"))
    for asset in AS:
        selected=[x for x in lock["selection"] if x["asset"]==asset]
        gmap={x["objective"]:float(x["gamma"]) for x in selected}
        gids={o:int(np.where(np.isclose(GG,g))[0][0]) for o,g in gmap.items()}
        ids=[i for i,x in enumerate(starts) if x[1]==asset]
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,620001,1 if asset=="BTCUSDT" else 2]))
            N=a.test_trajectories
            costs={o:np.empty(N,float) for o in OB}; comps={o:np.empty(N,float) for o in OB}
            for n in range(N):
                j=int(rng.choice(ids)); st,_,_=starts[j]
                th,_,_=d3.sample_reference_theta_path(df,st,120,ctx[3],ctx[4],ctx[5],rng,ctx[1],ctx[2])
                epseed=int(np.random.SeedSequence([seed,630001,n]).generate_state(1,dtype=np.uint64)[0])
                z=turbo_episode(d3,df.iloc[st],asset,th,epseed,ctx)
                for o in OB:
                    costs[o][n]=z["shortfall_bps"][gids[o]]
                    comps[o][n]=z["completion_rate"][gids[o]]
                if n==0 or (n+1)%1000==0: print(f"TURBO TEST {asset} seed={seed} {n+1}/{N}",flush=True)
            u=thresholds[asset]
            for o in OB:
                x=costs[o]; cv95,_=cvar(x,.95); cv99,_=cvar(x,.99)
                rows.append(dict(asset=asset,seed=seed,objective=o,gamma=gmap[o],n=N,
                    mean=float(x.mean()),cvar95=cv95,cvar99=cv99,
                    train_frozen_exceedance_threshold=u,
                    exceedance_rate=float(np.mean(x>u)),
                    mean_excess_given_exceedance=float(np.mean(x[x>u]-u)) if np.any(x>u) else 0.0,
                    completion=float(comps[o].mean())))
    T=pd.DataFrame(rows); T.to_csv(out/"TEST_SEED_METRICS.csv",index=False)
    brng=np.random.default_rng(640001); agg=[]
    for (asset,o),g in T.groupby(["asset","objective"]):
        for metric in ["mean","cvar95","cvar99","exceedance_rate","mean_excess_given_exceedance","completion"]:
            v=g.sort_values("seed")[metric].to_numpy(float); boots=np.empty(5000)
            for b in range(5000): boots[b]=brng.choice(v,size=len(v),replace=True).mean()
            agg.append(dict(asset=asset,objective=o,metric=metric,estimate=float(v.mean()),
                ci025=float(np.quantile(boots,.025)),ci975=float(np.quantile(boots,.975)),
                n_seeds=len(v),bootstrap_reps=5000))
    pd.DataFrame(agg).to_csv(out/"TEST_BOOTSTRAP_CI.csv",index=False)
    dump(out/"TEST_COMPLETE.json",{"version":"V1-I.6-TURBO.1-TEST","complete":True,
         "test_once":True,"n_per_policy_seed":a.test_trajectories,"seeds":SEEDS,
         "exceedance_threshold":"asset-specific TRAIN expected-cost-policy VaR95, frozen before TEST"})
    print("TURBO TEST COMPLETE",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["train","validation","test"],required=True)
    p.add_argument("--protocol",required=True); p.add_argument("--results",required=True)
    p.add_argument("--d3-script",required=True); p.add_argument("--h33-script",required=True)
    p.add_argument("--panel",required=True); p.add_argument("--vep-results",required=True)
    p.add_argument("--g55-results",required=True); p.add_argument("--h22-results",required=True)
    p.add_argument("--device",default="cuda"); p.add_argument("--test-trajectories",type=int,default=20000)
    p.add_argument("--jobs",type=int,default=4)
    a=p.parse_args()
    if js(a.protocol).get("design_sha256")!=HASH: raise SystemExit("protocol hash mismatch")
    if a.jobs<1 or a.jobs>8: raise SystemExit("--jobs must be in 1..8")
    out=Path(a.results); out.mkdir(parents=True,exist_ok=True)
    d3,h,raw,*rest=setup(a); ctx=(h,*rest)
    if a.stage=="train": train(a,d3,raw,ctx,out)
    elif a.stage=="validation": validation(a,d3,raw,ctx,out)
    else:
        if a.test_trajectories<20000: raise SystemExit("TEST N<20000 forbidden")
        test(a,d3,raw,ctx,out)

if __name__=="__main__":
    main()
