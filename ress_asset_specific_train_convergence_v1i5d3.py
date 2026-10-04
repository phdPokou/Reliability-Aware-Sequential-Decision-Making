#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RESS V1-I.5-D.1.1a — Factorized latent-path interface (TRAIN-only validation).

This is NOT yet the Localized-KL optimizer. It constructs the missing numerical
factorization required by the locked theory:

    P_Theta(dtheta) K_C^gamma(dc | theta).

P_Theta in this repaired development environment is built policy-independently
from frozen F.5.3 predictive priors evaluated on EXACT 5-second contiguous
OBSERVED TRAIN predictor histories ("teacher-forced reference paths"). The
ambiguous latent primitive has length exactly H=T. A sampled theta_1:H path is
held fixed during the decision horizon while independent conditional
market/execution randomness is replayed under the SAME observable policy.
Post-decision mechanical settlement remains inside K_C^gamma(.|theta_1:H):
any settlement latent scales are drawn from the frozen nominal predictive prior
and are not components of the KL-ambiguous theta path.

Important consequence:
This factorized environment is not identical to the recursive-theta H.3.3
environment used in I.2-I.4, where theta's predictive parameters are recomputed
from simulated evolving histories. Therefore I.2-I.4 must eventually be rerun
under the factorized environment before publication-level Localized-vs-Global
comparisons. This script does not silently claim comparability.

No TEST data. No KL robustification. No policy selection. Theta is never a
policy input.
"""
from pathlib import Path
import argparse, json, time, importlib.util, json, math, time
import numpy as np
import pandas as pd

EPS=1e-12

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def bind_h33(h33,i25):
    names=["XI","XFULL","reconstruct_depth","sample_book_shape","walk_bid",
           "forced_terminal_walk","center_std","inv_std","liq_center","spread_next"]
    missing=[n for n in names if not hasattr(h33,n)]
    if missing: raise RuntimeError(f"H.3.3 missing required symbols: {missing}")
    for n in names: globals()[n]=getattr(h33,n)
    return names

def sample_eps_given_theta(theta,rng,noise_cache):
    nu,R=noise_cache
    g=rng.multivariate_normal(np.zeros(3),R)
    u=rng.chisquare(nu)
    eps=g/math.sqrt(u/nu)
    return math.sqrt(float(theta))*eps

def simulate_policy_episode_fixed_theta(r0,sym,H,inv_frac,policy,max_participation,K,model,ln,sc,center_fits,
                                        liq,spread_spec,spread_resid,book,rng,torch,device,rv_window,
                                        noise_cache,max_terminal_steps,theta_path):
    mid=float(r0.mid); total=float(r0.total_depth); ratio=float(r0.log_depth_ratio)
    spread_bps=float(r0.spread_bps); imb=float(r0.imbalance)
    rv=max(float(r0.rv_backward),EPS)
    last=np.array([float(r0[c]) for c in XI])
    db,_,_=reconstruct_depth(total,ratio)
    q0=max(inv_frac*db,EPS); q=q0
    mid0=max(mid,EPS); depth0=max(total,EPS)
    sf=0.; filled_total=0.; theta_max=0.; theta_sum=0.; theta_n=0
    regular_sf=0.; terminal_sf=0.
    max_part=0.; numerical_fail=False; book_fallback_steps=0
    # Preserve the observed backward-RV scale at episode start, then roll it out
    # causally over rv_window simulated returns rather than resetting after one step.
    rv_window=max(int(rv_window),1)
    rets=[rv/math.sqrt(rv_window)]*rv_window
    for t in range(H):
        if not all(np.isfinite(v) for v in [mid,total,ratio,spread_bps,rv]) or min(mid,total,spread_bps,rv)<=0:
            numerical_fail=True; break
        db,da,imb=reconstruct_depth(total,ratio)
        spread=mid*spread_bps/1e4
        shares,mults,exact_cell,_=sample_book_shape(sym,spread_bps,total,imb,book,rng,K)
        if not exact_cell: book_fallback_steps+=1
        feat=np.array([
            q/max(q0,EPS),
            (H-t)/max(H,1),
            math.log(max(mid,EPS)/mid0),
            math.log(max(spread_bps,EPS)),
            math.log(max(total,EPS)/depth0),
            imb,
            math.log(max(rv,EPS)),
            last[0],last[1],last[2],
            1.0 if sym=="ETHUSDT" else 0.0
        ],dtype=np.float32)
        with torch.no_grad():
            xt=torch.as_tensor(feat,device=device).unsqueeze(0)
            frac=float(policy(xt).item())
        cap=min(float(q),max(float(max_participation),0.0)*max(float(db),0.0))
        a=cap*min(max(frac,0.0),1.0)
        fill,vwap,cost,_=walk_bid(mid,spread,db,a,shares,mults)
        q-=fill; filled_total+=fill; sf+=cost; regular_sf+=cost
        if db>0:max_part=max(max_part,fill/db)

        xraw=np.array([last[0],last[1],last[2],math.log(spread_bps),imb,math.log(rv)])
        xstd=np.array([(xraw[j]-float(sc[XFULL[j]]["mean"]))/
                       max(float(sc[XFULL[j]]["sd"]),EPS) for j in range(len(XFULL))])
        if theta_n >= len(theta_path):
            raise RuntimeError("Fixed theta path exhausted before trajectory termination")
        theta=float(theta_path[theta_n])
        if (not np.isfinite(theta)) or theta<=0:
            raise RuntimeError("Fixed theta path contains nonpositive/nonfinite value")
        e=sample_eps_given_theta(theta,rng,noise_cache)
        theta_max=max(theta_max,theta); theta_sum+=theta; theta_n+=1
        cstd=np.asarray(center_std(sym,xstd,center_fits),float)
        dmid=inv_std("dlog_mid",cstd[0]+e[0],sc)

        y=np.array([math.log(total),ratio])
        zliq=np.array([last[0],math.log(spread_bps),imb,math.log(rv)])
        lc=liq_center(liq[sym],y,zliq)
        rs=np.asarray(liq[sym]["resid_sd"],float)
        yn=lc+rs*np.asarray(e[1:3],float)
        total_n=math.exp(float(yn[0])); ratio_n=float(yn[1])
        mid_n=math.exp(math.log(mid)+dmid)

        ddepth=math.log(total_n)-math.log(total)
        dratio=ratio_n-ratio
        sf_feat={"imbalance":imb,"log_rv_now":math.log(rv),"dlog_mid":dmid,
                 "dlog_total_depth":ddepth,"d_log_depth_ratio":dratio}
        spread_n=spread_next(spread_spec[sym],spread_bps,sf_feat,rng,spread_resid[sym])

        rets.append(dmid); rets=rets[-rv_window:]
        rv_n=max(float(np.sqrt(np.sum(np.square(rets)))),EPS)
        mid,total,ratio,spread_bps,rv=mid_n,total_n,ratio_n,spread_n,rv_n
        last=np.array([dmid,ddepth,dratio])

    if theta_n != H:
        raise RuntimeError(f"Decision-horizon theta consumption mismatch: used {theta_n}, expected {H}")

    # Terminal settlement window.  The policy stops at H.
    # Residual inventory is settled mechanically through successive empirical
    # K-level books. Between books, the same frozen exogenous G.5.5 transition
    # advances once. No policy decision, no regular participation cap, and no K+
    # liquidity extrapolation are introduced during settlement.
    q_pre_terminal=max(float(q),0.0)
    forced_fill=0.; forced_cost=0.; settlement_steps=0
    terminal_book_fallback_steps=0
    settlement_numerical_failure=False

    while (q>1e-12) and (settlement_steps<max_terminal_steps) and (not numerical_fail):
        if not all(np.isfinite(v) for v in [mid,total,ratio,spread_bps,rv]) or min(mid,total,spread_bps,rv)<=0:
            settlement_numerical_failure=True
            break

        db,da,imb=reconstruct_depth(total,ratio)
        spread=mid*spread_bps/1e4
        shares,mults,exact_cell,_=sample_book_shape(sym,spread_bps,total,imb,book,rng,K)
        if not exact_cell: terminal_book_fallback_steps+=1

        f,forced_vwap,cst,residual=forced_terminal_walk(mid,spread,db,q,shares,mults)
        forced_fill+=f; forced_cost+=cst; sf+=cst; filled_total+=f
        q=residual
        settlement_steps+=1
        if q<=1e-12:
            q=0.0
            break

        # Same frozen exogenous transition as in the controlled horizon.
        xraw=np.array([last[0],last[1],last[2],math.log(spread_bps),imb,math.log(rv)])
        xstd=np.array([(xraw[j]-float(sc[XFULL[j]]["mean"]))/
                       max(float(sc[XFULL[j]]["sd"]),EPS) for j in range(len(XFULL))])
        # Post-decision settlement belongs to K_C^pi(.|theta_1:H), not to
        # the ambiguous primitive. Draw its latent scale from the frozen nominal
        # F.5.3 prior conditional on the evolving observable predictor.
        with torch.inference_mode():
            xt_prior=torch.as_tensor(np.asarray(xstd,np.float32)[None,:],device=device)
            mt,st=model.prior.moments(xt_prior,ln)
            mpost=float(mt.detach().cpu()[0]); spost=float(st.detach().cpu()[0])
        theta=math.exp(rng.normal(mpost,spost))
        if (not np.isfinite(theta)) or theta<=0:
            raise RuntimeError("Nominal post-decision theta draw nonpositive/nonfinite")
        e=sample_eps_given_theta(theta,rng,noise_cache)
        # Do not include nominal settlement scales in summaries of the ambiguous
        # decision-horizon theta path.
        cstd=np.asarray(center_std(sym,xstd,center_fits),float)
        dmid=inv_std("dlog_mid",cstd[0]+e[0],sc)

        y=np.array([math.log(total),ratio])
        zliq=np.array([last[0],math.log(spread_bps),imb,math.log(rv)])
        lc=liq_center(liq[sym],y,zliq)
        rs=np.asarray(liq[sym]["resid_sd"],float)
        yn=lc+rs*np.asarray(e[1:3],float)
        total_n=math.exp(float(yn[0])); ratio_n=float(yn[1])
        mid_n=math.exp(math.log(mid)+dmid)

        ddepth=math.log(total_n)-math.log(total)
        dratio=ratio_n-ratio
        sf_feat={"imbalance":imb,"log_rv_now":math.log(rv),"dlog_mid":dmid,
                 "dlog_total_depth":ddepth,"d_log_depth_ratio":dratio}
        spread_n=spread_next(spread_spec[sym],spread_bps,sf_feat,rng,spread_resid[sym])
        rets.append(dmid); rets=rets[-rv_window:]
        rv_n=max(float(np.sqrt(np.sum(np.square(rets)))),EPS)
        mid,total,ratio,spread_bps,rv=mid_n,total_n,ratio_n,spread_n,rv_n
        last=np.array([dmid,ddepth,dratio])

    terminal_sf=forced_cost
    forced_residual=max(float(q),0.0)
    terminal_exhausted=bool(forced_residual>1e-12 and settlement_steps>=max_terminal_steps)
    terminal_book_fallback_rate=terminal_book_fallback_steps/max(settlement_steps,1)

    arrival=float(r0.mid)
    regular_completion=(q0-q_pre_terminal)/q0
    final_completion=filled_total/q0
    sbps=1e4*sf/(arrival*q0) if arrival>0 else np.nan
    regular_bps=1e4*regular_sf/(arrival*q0) if arrival>0 else np.nan
    terminal_bps=1e4*terminal_sf/(arrival*q0) if arrival>0 else np.nan
    return {"q0":q0,
            "regular_filled":q0-q_pre_terminal,
            "forced_terminal_fill":forced_fill,
            "filled":filled_total,
            "pre_terminal_inventory":q_pre_terminal,
            "terminal_inventory":q,
            "regular_completion_rate":regular_completion,
            "completion_rate":final_completion,
            "terminal_liquidation_required":bool(q_pre_terminal>1e-12),
            "terminal_settlement_steps":int(settlement_steps),
            "terminal_settlement_exhausted":terminal_exhausted,
            "terminal_settlement_numerical_failure":settlement_numerical_failure,
            "terminal_unliquidated_fraction":forced_residual/q0,
            "regular_shortfall_bps":regular_bps,
            "terminal_forced_shortfall_bps":terminal_bps,
            "shortfall_bps":sbps,
            "regular_implementation_shortfall":regular_sf,
            "terminal_forced_cost":terminal_sf,
            "implementation_shortfall":sf,
            "max_step_participation":max_part,
            "book_fallback_step_rate":book_fallback_steps/max(H,1),
            "terminal_book_fallback_step_rate":terminal_book_fallback_rate,
            "theta_mean_ex_post":theta_sum/max(theta_n,1),
            "theta_max_ex_post":theta_max,
            "simulated_steps":theta_n,
            "numerical_failure":bool(numerical_fail or settlement_numerical_failure)}

def observed_predictor_xstd(row,sc):
    """
    Exact F.5.3 predictive feature interface.
    XFULL = [lag_dlog_mid, lag_dlog_total_depth, lag_d_log_depth_ratio,
             lag_log_spread, lag_imbalance, lag_log_rv].
    These columns are already constructed causally by H.3.3 build_raw().
    """
    vals=np.array([float(row[c]) for c in XFULL],float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("Non-finite frozen F.5.3 predictor row")
    return np.array([(vals[j]-float(sc[XFULL[j]]["mean"]))/
                     max(float(sc[XFULL[j]]["sd"]),EPS) for j in range(len(XFULL))],float)

def _timestamp_seconds(s, expected_grid_seconds):
    """
    Convert grid_timestamp to seconds without assuming its numeric unit.

    V1-D stores grid_timestamp in the provider/grid representation. Numeric
    timestamps may therefore be seconds, milliseconds, microseconds, or
    nanoseconds. Infer the unit from within-day adjacent positive gaps relative
    to the known analytical grid; fail closed if no standard unit is compatible.
    """
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().all():
        v=x.to_numpy(float)
        dv=np.diff(v)
        pos=dv[np.isfinite(dv) & (dv>0)]
        if len(pos)==0:
            raise RuntimeError("Cannot infer numeric grid_timestamp unit: no positive gaps")
        med=float(np.median(pos))
        units=np.array([1.0,1e3,1e6,1e9],float)  # numeric units per second
        target=float(expected_grid_seconds)
        err=np.abs(med/units-target)/max(abs(target),1e-12)
        j=int(np.argmin(err))
        if err[j] > 1e-6:
            raise RuntimeError(
                f"Cannot map numeric grid_timestamp to {target:g}s grid: "
                f"median_raw_gap={med:.12g}, best_units_per_second={units[j]:.0e}, "
                f"relative_error={err[j]:.3e}")
        return v/units[j], float(units[j])
    dt=pd.to_datetime(s,errors="coerce",utc=True)
    if dt.isna().any():
        raise RuntimeError("grid_timestamp cannot be parsed as numeric or datetime")
    return dt.astype("int64").to_numpy(float)/1e9, 1e9

def build_train_windows(raw,path_len,grid_seconds):
    """
    Construct TRAIN windows with an explicit exact-grid gate.
    A candidate window is admissible only if every adjacent timestamp differs
    from grid_seconds (within floating representation tolerance). dropna may
    therefore shorten/split runs but can never bridge a temporal gap.
    """
    d=raw[raw.split.astype(str).str.lower().eq("train")].copy()
    need=["symbol","date","grid_timestamp","dlog_mid","dlog_total_depth",
          "d_log_depth_ratio","spread_bps","imbalance","rv_backward",
          "mid","total_depth","log_depth_ratio"] + list(XFULL)
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=need)
    d=d.sort_values(["symbol","date","grid_timestamp"]).reset_index(drop=True)
    starts=[]
    tol=max(1e-9,abs(float(grid_seconds))*1e-9)
    for (_, _),g in d.groupby(["symbol","date"],sort=False):
        ids=g.index.to_numpy()
        ts,ts_units_per_second=_timestamp_seconds(g["grid_timestamp"],grid_seconds)
        if len(ids)<path_len: continue
        gaps=np.diff(ts)
        good=np.isclose(gaps,float(grid_seconds),rtol=0.0,atol=tol)
        # Split at every non-grid gap. Candidate windows never cross a split.
        run_start=0
        boundaries=np.flatnonzero(~good)+1
        for run_end in list(boundaries)+[len(ids)]:
            run_ids=ids[run_start:run_end]
            if len(run_ids)>=path_len:
                stride=max(path_len//4,1)
                for k in range(0,len(run_ids)-path_len+1,stride):
                    s=int(run_ids[k])
                    # Defensive exact-window verification.
                    w=d.iloc[s:s+path_len]
                    if len(w)!=path_len: raise RuntimeError("Window indexing failure")
                    wt,_=_timestamp_seconds(w["grid_timestamp"],grid_seconds)
                    if not np.all(np.isclose(np.diff(wt),float(grid_seconds),rtol=0.0,atol=tol)):
                        raise RuntimeError("Internal contiguity invariant violated")
                    starts.append((s,str(w.iloc[0].symbol),str(w.iloc[0].date)))
            run_start=run_end
    if not starts:
        raise RuntimeError(f"No exact-{grid_seconds:g}s contiguous TRAIN window of length {path_len}")
    # Record conversion diagnostics for the caller. All groups must map to a
    # standard time unit; exact-window checks above remain the admissibility gate.
    d.attrs["grid_seconds_verified"]=float(grid_seconds)
    d.attrs["n_admissible_starts"]=int(len(starts))
    return d,starts

def sample_reference_theta_path(train_df,start,path_len,model,ln,sc,rng,torch,device):
    theta=np.empty(path_len,float); mu=np.empty(path_len,float); sd=np.empty(path_len,float)
    # build_train_windows guarantees contiguous rows within the same symbol/date.
    for t in range(path_len):
        row=train_df.iloc[start+t]
        xstd=observed_predictor_xstd(row,sc)
        with torch.inference_mode():
            xt=torch.as_tensor(np.asarray(xstd,np.float32)[None,:],device=device)
            mt,st=model.prior.moments(xt,ln)
            m=float(mt.detach().cpu()[0]); s=float(st.detach().cpu()[0])
        z=rng.normal(m,s)
        theta[t]=math.exp(z); mu[t]=m; sd[t]=s
    return theta,mu,sd

class GammaPolicy:
    def __init__(self,gamma,torch,device):
        self.gamma=float(gamma); self.torch=torch; self.device=device
    def __call__(self,x):
        return self.torch.full((x.shape[0],1),self.gamma,dtype=x.dtype,device=x.device)
    def eval(self): return self


def _kl_sup_weights(G, epsilon):
    """Exact finite empirical KL-ball supremum for uniform reference atoms."""
    G=np.asarray(G,float)
    M=len(G)
    if M<2 or not np.isfinite(G).all(): raise ValueError("Invalid G")
    if epsilon<=1e-15:
        return float(G.mean()), np.full(M,1.0/M), np.inf, 0.0
    imax=int(np.argmax(G))
    # A point mass is feasible iff epsilon >= log M.
    if epsilon >= math.log(M)-1e-14:
        q=np.zeros(M); q[imax]=1.0
        return float(G[imax]),q,0.0,float(math.log(M))
    def weights(lam):
        z=G/lam; z-=z.max()
        w=np.exp(z); w/=w.sum()
        return w
    def kl_of(lam):
        q=weights(lam)
        return float(np.sum(q*np.log(np.maximum(q*M,1e-300))))
    lo=1e-14; hi=max(float(np.std(G)),float(np.max(np.abs(G))),1e-8)
    while kl_of(hi)>epsilon: hi*=2.0
    for _ in range(120):
        mid=math.sqrt(lo*hi)
        if kl_of(mid)>epsilon: lo=mid
        else: hi=mid
    lam=hi; q=weights(lam); kl=kl_of(lam)
    return float(q@G),q,float(lam),kl

def localized_ru(C, alpha, epsilon):
    """Finite nested localized RU objective; eta optimized over empirical breakpoints."""
    C=np.asarray(C,float)
    if C.ndim!=2 or not np.isfinite(C).all(): raise ValueError("C must be finite MxL")
    vals=np.unique(C.ravel())
    # Convex piecewise-linear-in-eta outer envelope: empirical breakpoints suffice.
    best=None
    for eta in vals:
        G=np.maximum(C-eta,0.0).mean(axis=1)
        sup,q,lam,kl=_kl_sup_weights(G,epsilon)
        rho=float(eta+sup/(1-alpha))
        if best is None or rho<best["rho"]:
            best={"rho":rho,"eta":float(eta),"G":G,"q":q,"lambda":lam,"achieved_kl":kl}
    return best

def _seed(master, asset_code, outer_id, inner_id):
    ss=np.random.SeedSequence([int(master),310003,int(asset_code),int(outer_id),int(inner_id)])
    x=int(ss.generate_state(1,dtype=np.uint64)[0] % np.uint64(2**63-1))
    return max(x,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h33-script",default="ress_coupled_execution_v1h33.py")
    ap.add_argument("--panel",default="Results_RESS_V1D/panels")
    ap.add_argument("--vep-results",default="Results_RESS_V1F53")
    ap.add_argument("--g55-results",default="Results_RESS_V1G55")
    ap.add_argument("--h22-results",default="Results_RESS_V1H22")
    ap.add_argument("--protocol",default="Results_RESS_V1I5D2/V1I5D2_PROTOCOL.json")
    ap.add_argument("--results",default="Results_RESS_V1I5D3")
    ap.add_argument("--grid-seconds",type=int,default=5)
    ap.add_argument("--K",type=int,default=10)
    ap.add_argument("--horizon",type=int,default=120)
    ap.add_argument("--inventory-fraction",type=float,default=2.0)
    ap.add_argument("--max-participation",type=float,default=.10)
    ap.add_argument("--max-terminal-steps",type=int,default=960)
    ap.add_argument("--rv-window",type=int,default=12)
    ap.add_argument("--gamma",type=float,default=.5)
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    protocol=json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    expected_hash="0791be105f4afd18170c7e5b12a74078afc684815414f713ecb80f46a99bcf50"
    if protocol.get("design_sha256")!=expected_hash:
        raise RuntimeError("D.2 protocol hash mismatch")
    if protocol["inner_convergence"]["L_ladder"] != [512,1024,2048]: raise RuntimeError("L ladder mismatch")
    if protocol["outer_convergence"]["M_ladder"] != [64,128,256]: raise RuntimeError("M ladder mismatch")
    if abs(float(protocol["inner_convergence"]["relative_tolerance"])-.05)>1e-15: raise RuntimeError("L tolerance mismatch")
    if abs(float(protocol["outer_convergence"]["relative_tolerance"])-.05)>1e-15: raise RuntimeError("M tolerance mismatch")
    if abs(a.gamma-.5)>1e-15: raise ValueError("D.3 convergence gate locked to gamma=.5")
    if a.horizon!=120 or a.grid_seconds!=5 or a.max_terminal_steps!=960: raise ValueError("D.1.1a design mismatch")

    Mmax=256; Lmax=2048; eps=.08; alpha=.95
    out=Path(a.results); (out/"tables").mkdir(parents=True,exist_ok=True)
    t0=time.time()

    h33=load_module(Path(a.h33_script),"h33_i5d3")
    class Dummy: pass
    bind_h33(h33,Dummy)
    raw,_=h33.load_panel(a.panel); raw=h33.build_raw(raw,a.grid_seconds,a.K)
    train_df,starts=build_train_windows(raw,a.horizon,a.grid_seconds)
    print(f"EXACT GRID WINDOWS PASS | grid={a.grid_seconds}s | H={a.horizon} | candidate_starts={len(starts):,}",flush=True)

    liq=h33.load_json(Path(a.g55_results)/"models"/"v1g55_liquidity_varx.json")
    spr=h33.load_json(Path(a.g55_results)/"models"/"v1g55_spread_arx.json")
    sres=h33.load_npz_resid(Path(a.g55_results)/"models"/"v1g55_spread_residuals.npz")
    book=h33.load_empirical_book(a.h22_results,a.K)
    import torch, torch.nn as nn, torch.nn.functional as F
    device=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    model,xcols,ln,sc,center_fits,cfg=h33.load_frozen_m2(a.vep_results,device,torch,nn,F)
    if xcols != h33.XFULL: raise RuntimeError("Frozen XFULL mismatch")
    with torch.inference_mode():
        nu_t,R_t=model.noise()
        noise_cache=(float(nu_t.detach().cpu()),R_t.detach().cpu().numpy())

    # Asset-specific admissible starts; sample Mmax outer atoms independently by asset.
    by_asset={}
    for asset in ("BTCUSDT","ETHUSDT"):
        ids=[i for i,x in enumerate(starts) if str(x[1])==asset]
        if len(ids)<Mmax:
            raise RuntimeError(f"{asset}: only {len(ids)} admissible starts; need >= {Mmax} without replacement")
        by_asset[asset]=ids

    policy=GammaPolicy(a.gamma,torch,device)
    all_results={}
    for asset_code,asset in enumerate(("BTCUSDT","ETHUSDT"),start=1):
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,310001,asset_code]))
        chosen_local=rng.choice(by_asset[asset],size=Mmax,replace=False)
        theta_bank=[]; outer_rows=[]
        for m,jj in enumerate(chosen_local):
            start,sym,date=starts[int(jj)]
            theta,mu,sd=sample_reference_theta_path(train_df,start,a.horizon,model,ln,sc,rng,torch,device)
            theta_bank.append((start,theta))
            outer_rows.append({"asset":asset,"outer_id":m,"start_index":int(start),"date":str(date),
              "theta_mean":float(theta.mean()),"theta_sd":float(theta.std(ddof=1)),
              "theta_min":float(theta.min()),"theta_max":float(theta.max())})
        pd.DataFrame(outer_rows).to_csv(out/"tables"/f"v1i5d3_{asset}_outer_atoms.csv",index=False)

        C=np.empty((Mmax,Lmax),float)
        for m,(start,theta) in enumerate(theta_bank):
            r0=train_df.iloc[start]
            for l in range(Lmax):
                sd=_seed(a.seed,asset_code,m,l)
                z=simulate_policy_episode_fixed_theta(
                    r0,asset,a.horizon,a.inventory_fraction,policy,a.max_participation,a.K,
                    model,ln,sc,center_fits,liq,spr,sres,book,np.random.default_rng(sd),
                    torch,device,a.rv_window,noise_cache,a.max_terminal_steps,theta)
                if z["numerical_failure"] or z["terminal_settlement_exhausted"]:
                    raise RuntimeError(f"{asset}: mechanical failure outer={m} inner={l}")
                if abs(float(z["theta_mean_ex_post"])-float(theta.mean()))>1e-10:
                    raise RuntimeError(f"{asset}: fixed-theta mean mismatch outer={m} inner={l}")
                C[m,l]=float(z["shortfall_bps"])
            if (m+1)%8==0 or m==0:
                print(f"{asset} OUTER {m+1}/{Mmax} | completed L={Lmax}",flush=True)
        np.save(out/"tables"/f"v1i5d3_{asset}_cost_matrix_M256_L2048.npy",C)

        # Inner gate at fixed M=256, nested L prefixes.
        inner=[]
        for L in (512,1024,2048):
            rr=localized_ru(C[:,:L],alpha,eps)
            inner.append({"asset":asset,"M":Mmax,"L":L,"epsilon":eps,"rho":rr["rho"],
                          "eta":rr["eta"],"lambda":rr["lambda"],"achieved_kl":rr["achieved_kl"],
                          "neff":float(1/np.sum(rr["q"]**2)),"qmax":float(rr["q"].max())})
        inner[-1]["relative_change_vs_previous"]=abs(inner[-1]["rho"]-inner[-2]["rho"])/max(abs(inner[-1]["rho"]),1e-18)
        inner_pass=bool(inner[-1]["relative_change_vs_previous"]<=.05)

        # Outer gate at fixed L=2048, nested M prefixes.
        outer=[]
        for M in (64,128,256):
            rr=localized_ru(C[:M,:],alpha,eps)
            contrib=rr["q"]*rr["G"]; order=np.argsort(-contrib); total=float(contrib.sum())
            sh=lambda k: float(contrib[order[:k]].sum()/total) if abs(total)>1e-18 else None
            outer.append({"asset":asset,"M":M,"L":Lmax,"epsilon":eps,"rho":rr["rho"],
                          "eta":rr["eta"],"lambda":rr["lambda"],"achieved_kl":rr["achieved_kl"],
                          "neff":float(1/np.sum(rr["q"]**2)),"qmax":float(rr["q"].max()),
                          "top1_share":sh(1),"top5_share":sh(5),"top10_share":sh(10)})
        outer[-1]["relative_change_vs_previous"]=abs(outer[-1]["rho"]-outer[-2]["rho"])/max(abs(outer[-1]["rho"]),1e-18)
        outer_pass=bool(outer[-1]["relative_change_vs_previous"]<=.05)

        pd.DataFrame(inner).to_csv(out/"tables"/f"v1i5d3_{asset}_inner_gate.csv",index=False)
        pd.DataFrame(outer).to_csv(out/"tables"/f"v1i5d3_{asset}_outer_gate.csv",index=False)
        all_results[asset]={"inner_gate":inner,"inner_pass":inner_pass,
                            "outer_gate":outer,"outer_pass":outer_pass,
                            "ready_for_gamma_selection":bool(inner_pass and outer_pass)}
        print(f"{asset} | INNER PASS={inner_pass} | OUTER PASS={outer_pass}",flush=True)

    summary={"version":"V1-I.5-D.3","role":"asset-specific TRAIN numerical convergence",
      "protocol_sha256":expected_hash,"gamma":a.gamma,"alpha":alpha,"epsilon_gate":eps,
      "M_ladder":[64,128,256],"L_ladder":[512,1024,2048],
      "results":all_results,
      "all_assets_ready_for_gamma_selection":bool(all(v["ready_for_gamma_selection"] for v in all_results.values())),
      "guards":{"train_only":True,"no_validation":True,"no_test":True,
                "no_policy_selection":True,"no_radius_selection":True,
                "assets_not_pooled":True,"common_nested_prefixes":True},
      "elapsed_seconds":time.time()-t0}
    (out/"V1I5D3_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("\nNO VALIDATION | NO TEST | NO POLICY/RADIUS SELECTION | ASSETS SEPARATE")
    print(f"DONE | V1-I.5-D.3 | Results -> {out.resolve()}")

if __name__=="__main__":
    main()
