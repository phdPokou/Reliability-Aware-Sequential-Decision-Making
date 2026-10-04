#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RESS V1-I.6-FAST worker. Long jobs are checkpointed by asset/gamma/seed/policy."""
import os
# Must be set before NumPy/SciPy/Torch/loaded scientific modules.
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
import argparse,importlib.util,json,math,time,threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
HASH="0791be105f4afd18170c7e5b12a74078afc684815414f713ecb80f46a99bcf50"
AS=("BTCUSDT","ETHUSDT"); OB=("expected_cost","nominal_cvar","global_kl","localized_kl")
GG=np.round(np.linspace(0,1,11),1); EPS=[0.,.02,.04,.08,.12,.16]
SEEDS=[101,202,303,404,505,606,707,808,909,1001,1111,1212,1313,1414,1515]
def lm(p,n):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def js(p): return json.loads(Path(p).read_text())
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2))
def split_windows(d3,raw,split,H,grid):
 z=raw.copy(); z["split"]=np.where(z["split"].astype(str).str.lower().eq(split),"train","excluded")
 return d3.build_train_windows(z,H,grid)
def cvar(x,a=.95):
 x=np.asarray(x,float).ravel(); q=float(np.quantile(x,a,method="lower"))
 return float(q+np.maximum(x-q,0).mean()/(1-a)),q
def klweights(G,e,d3): return d3._kl_sup_weights(np.asarray(G,float),e)
def loc_fast(C,a,e,d3):
 C=np.asarray(C,float); lo=float(C.min()); hi=float(C.max())
 def f(eta):
  G=np.maximum(C-eta,0).mean(1); sup,_,_,_=klweights(G,e,d3); return float(eta+sup/(1-a))
 o=minimize_scalar(f,bounds=(lo,hi),method="bounded",options={"xatol":max(1e-13,(hi-lo)*1e-10),"maxiter":100})
 eta=float(o.x); G=np.maximum(C-eta,0).mean(1); sup,q,lam,kl=klweights(G,e,d3)
 return dict(rho=float(eta+sup/(1-a)),eta=eta,lambda_=lam,kl=kl,neff=float(1/(q@q)),qmax=float(q.max()))
def glob_fast(x,a,e,d3):
 x=np.asarray(x,float).ravel(); lo=float(x.min()); hi=float(x.max())
 def f(eta):
  G=np.maximum(x-eta,0); sup,_,_,_=klweights(G,e,d3); return float(eta+sup/(1-a))
 o=minimize_scalar(f,bounds=(lo,hi),method="bounded",options={"xatol":max(1e-13,(hi-lo)*1e-10),"maxiter":100})
 eta=float(o.x); G=np.maximum(x-eta,0); sup,q,lam,kl=klweights(G,e,d3)
 return dict(rho=float(eta+sup/(1-a)),eta=eta,lambda_=lam,kl=kl,neff=float(1/(q@q)),qmax=float(q.max()))
def setup(a):
 d3=lm(a.d3_script,"d3_i6"); h=lm(a.h33_script,"h33_i6"); d3.bind_h33(h,type("D",(),{}))
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
def episode(d3,r0,asset,policy,theta,seed,ctx):
 h,torch,dev,model,ln,sc,cf,liq,spr,sres,book,noise=ctx
 z=d3.simulate_policy_episode_fixed_theta(r0,asset,120,2.,policy,.1,10,model,ln,sc,cf,liq,spr,sres,book,
   np.random.default_rng(int(seed)),torch,dev,12,noise,960,theta)
 if z["numerical_failure"] or z["terminal_settlement_exhausted"]: raise RuntimeError("episode failure")
 return z
def theta_bank(d3,df,starts,asset,M,seed,ctx):
 h,torch,dev,model,ln,sc,*_=ctx; ids=[i for i,x in enumerate(starts) if x[1]==asset]
 rng=np.random.default_rng(np.random.SeedSequence([seed,610001,1 if asset=="BTCUSDT" else 2]))
 ch=rng.choice(ids,size=M,replace=False); B=[]
 for j in ch:
  st,_,date=starts[int(j)]; th,_,_=d3.sample_reference_theta_path(df,st,120,model,ln,sc,rng,torch,dev); B.append((st,th,date))
 return B
def train(a,d3,raw,ctx,out):
 df,starts=split_windows(d3,raw,"train",120,5); rows=[]
 for asset in AS:
  M=512 if asset=="BTCUSDT" else 256; L=2048; B=theta_bank(d3,df,starts,asset,M,101,ctx)
  # Mandatory serial-vs-threaded deterministic equivalence gate before expensive work.
  if a.jobs>1:
   st0,th0,_=B[0]; r00=df.iloc[st0]; pol0=d3.GammaPolicy(.5,ctx[1],ctx[2]); ac0=1 if asset=="BTCUSDT" else 2
   seeds0=[d3._seed(101,ac0,0,l) for l in range(min(4,L))]
   serial=np.array([episode(d3,r00,asset,pol0,th0,z,ctx)["shortfall_bps"] for z in seeds0])
   with ThreadPoolExecutor(max_workers=min(a.jobs,len(seeds0))) as ex:
    threaded=np.array(list(ex.map(lambda z: episode(d3,r00,asset,pol0,th0,z,ctx)["shortfall_bps"],seeds0)))
   err=float(np.max(np.abs(serial-threaded)))
   if not np.allclose(serial,threaded,rtol=1e-10,atol=1e-12): raise RuntimeError(f"FAST equivalence gate FAIL {asset}: maxerr={err}")
   print(f"FAST EQUIVALENCE PASS | {asset} | jobs={a.jobs} | maxerr={err:.3e}",flush=True)
  for gamma in GG:
   fp=out/"train"/f"{asset}_g{gamma:.1f}.npy"; fp.parent.mkdir(parents=True,exist_ok=True)
   if fp.exists(): C=np.load(fp); print("REUSE",fp,flush=True)
   else:
    pol=d3.GammaPolicy(float(gamma),ctx[1],ctx[2]); C=np.empty((M,L))
    ac=1 if asset=="BTCUSDT" else 2
    def one_outer(m):
     st,th,_=B[m]; row=np.empty(L,float); r0=df.iloc[st]
     for l in range(L): row[l]=episode(d3,r0,asset,pol,th,d3._seed(101,ac,m,l),ctx)["shortfall_bps"]
     return m,row
    done=0
    if a.jobs==1:
     iterator=(one_outer(m) for m in range(M))
     for m,row in iterator:
      C[m]=row; done+=1
      if done==1 or done%8==0 or done==M: print(f"TRAIN {asset} g={gamma:.1f} {done}/{M}",flush=True)
    else:
     with ThreadPoolExecutor(max_workers=a.jobs) as ex:
      fut=[ex.submit(one_outer,m) for m in range(M)]
      for f in as_completed(fut):
       m,row=f.result(); C[m]=row; done+=1
       if done==1 or done%8==0 or done==M: print(f"TRAIN {asset} g={gamma:.1f} {done}/{M} | jobs={a.jobs}",flush=True)
    np.save(fp,C)
   ec=float(C.mean()); cv,eta=cvar(C); loc=loc_fast(C,.95,.08,d3); glo=glob_fast(C,.95,.08,d3)
   for o,v in [("expected_cost",ec),("nominal_cvar",cv),("global_kl",glo["rho"]),("localized_kl",loc["rho"])]:
    rows.append(dict(asset=asset,gamma=gamma,objective=o,value=v))
 R=pd.DataFrame(rows); R.to_csv(out/"train_objectives.csv",index=False)
 sel=[]
 for (asset,o),g in R.groupby(["asset","objective"]):
  z=g.sort_values(["value","gamma"]).iloc[0]; sel.append(dict(asset=asset,objective=o,gamma=float(z.gamma),train_value=float(z.value)))
 pd.DataFrame(sel).to_csv(out/"TRAIN_SELECTION.csv",index=False)
 # Full predeclared radius sensitivity, no new simulation and no radius selection.
 sens=[]
 for z in sel:
  asset,o,gamma=z["asset"],z["objective"],float(z["gamma"])
  C=np.load(out/"train"/f"{asset}_g{gamma:.1f}.npy")
  cv,_=cvar(C)
  for e in EPS:
   loc=loc_fast(C,.95,e,d3) if e>0 else {"rho":cv}
   glo=glob_fast(C,.95,e,d3) if e>0 else {"rho":cv}
   sens.append(dict(asset=asset,selected_for=o,gamma=gamma,epsilon=e,
                    nominal_cvar=cv,localized_kl=loc["rho"],global_kl=glo["rho"]))
 pd.DataFrame(sens).to_csv(out/"TRAIN_RADIUS_SENSITIVITY.csv",index=False)
 dump(out/"TRAIN_LOCK.json",{"version":"V1-I.6-TRAIN","locked":True,"selection":sel,"BTC_D3_outer_gate":"FAIL retained","no_validation":True,"no_test":True})
 print("TRAIN LOCKED",flush=True)
def validation(a,d3,raw,ctx,out):
 lock=js(out/"TRAIN_LOCK.json"); df,starts=split_windows(d3,raw,"validation",120,5); rows=[]
 for asset in AS:
  gs=sorted({float(x["gamma"]) for x in lock["selection"] if x["asset"]==asset}); B=theta_bank(d3,df,starts,asset,64,202,ctx)
  for gamma in gs:
   pol=d3.GammaPolicy(gamma,ctx[1],ctx[2]); C=np.empty((64,512)); ac=1 if asset=="BTCUSDT" else 2
   def one_val_outer(m):
    st,th,_=B[m]; row=np.empty(512,float); r0=df.iloc[st]
    for l in range(512): row[l]=episode(d3,r0,asset,pol,th,d3._seed(202,ac,m,l),ctx)["shortfall_bps"]
    return m,row
   if a.jobs==1:
    for m in range(64): mm,row=one_val_outer(m); C[mm]=row
   else:
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
     for f in as_completed([ex.submit(one_val_outer,m) for m in range(64)]):
      mm,row=f.result(); C[mm]=row
   ec=float(C.mean()); cv,_=cvar(C); loc=loc_fast(C,.95,.08,d3); glo=glob_fast(C,.95,.08,d3)
   vals={"expected_cost":ec,"nominal_cvar":cv,"global_kl":glo["rho"],"localized_kl":loc["rho"]}
   for o in OB:
    if any(x["asset"]==asset and x["objective"]==o and float(x["gamma"])==gamma for x in lock["selection"]):
     rows.append(dict(asset=asset,objective=o,locked_gamma=gamma,validation_value=vals[o]))
 V=pd.DataFrame(rows); V.to_csv(out/"VALIDATION_CONFIRMATION.csv",index=False)
 dump(out/"VALIDATION_LOCK.json",{"version":"V1-I.6-VALIDATION","confirmed_without_retuning":True,"rows":rows,"no_test":True})
 print("VALIDATION LOCKED | NO RETUNING",flush=True)
def test(a,d3,raw,ctx,out):
 lock=js(out/"TRAIN_LOCK.json"); df,starts=split_windows(d3,raw,"test",120,5); rows=[]
 # TEST: flat nominal OOS reliability distribution; 20k+ trajectories per policy x seed.
 for asset in AS:
  selected=[x for x in lock["selection"] if x["asset"]==asset]
  for seed in SEEDS:
   ids=[i for i,x in enumerate(starts) if x[1]==asset]; rng=np.random.default_rng(np.random.SeedSequence([seed,620001,1 if asset=="BTCUSDT" else 2]))
   # CRN: same sampled start/theta/episode seed sequence across selected policies.
   N=a.test_trajectories; costs={o:np.empty(N) for o in OB}; completions={o:np.empty(N) for o in OB}
   policies={x["objective"]:d3.GammaPolicy(float(x["gamma"]),ctx[1],ctx[2]) for x in selected}
   for n in range(N):
    j=int(rng.choice(ids)); st,_,_=starts[j]
    th,_,_=d3.sample_reference_theta_path(df,st,120,ctx[3],ctx[4],ctx[5],rng,ctx[1],ctx[2])
    epseed=int(np.random.SeedSequence([seed,630001,n]).generate_state(1,dtype=np.uint64)[0])
    for o in OB:
     z=episode(d3,df.iloc[st],asset,policies[o],th,epseed,ctx); costs[o][n]=z["shortfall_bps"]; completions[o][n]=z["completion_rate"]
    if n==0 or (n+1)%1000==0: print(f"TEST {asset} seed={seed} {n+1}/{N}",flush=True)
   for o in OB:
    x=costs[o]; cv95,_=cvar(x,.95); cv99,_=cvar(x,.99); u=float(np.quantile(x,.95))
    rows.append(dict(asset=asset,seed=seed,objective=o,gamma=float(policies[o].gamma) if hasattr(policies[o],"gamma") else float(next(z["gamma"] for z in selected if z["objective"]==o)),
      n=N,mean=float(x.mean()),cvar95=cv95,cvar99=cv99,q95=u,exceedance_rate=float(np.mean(x>u)),completion=float(completions[o].mean())))
 T=pd.DataFrame(rows); T.to_csv(out/"TEST_SEED_METRICS.csv",index=False)
 # Seed-level nonparametric bootstrap CIs; seeds, not individual trajectories, are resampled.
 brng=np.random.default_rng(640001); agg=[]
 for (asset,o),g in T.groupby(["asset","objective"]):
  for metric in ["mean","cvar95","cvar99","completion"]:
   v=g.sort_values("seed")[metric].to_numpy(float); boots=np.empty(5000)
   for b in range(5000): boots[b]=brng.choice(v,size=len(v),replace=True).mean()
   agg.append(dict(asset=asset,objective=o,metric=metric,estimate=float(v.mean()),
                   ci025=float(np.quantile(boots,.025)),ci975=float(np.quantile(boots,.975)),
                   n_seeds=len(v),bootstrap_reps=5000))
 pd.DataFrame(agg).to_csv(out/"TEST_BOOTSTRAP_CI.csv",index=False)
 dump(out/"TEST_COMPLETE.json",{"version":"V1-I.6-TEST","complete":True,"test_once":True,"n_per_policy_seed":a.test_trajectories,"seeds":SEEDS})
 print("TEST COMPLETE",flush=True)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--stage",choices=["train","validation","test"],required=True)
 p.add_argument("--protocol",required=True); p.add_argument("--results",required=True); p.add_argument("--d3-script",required=True); p.add_argument("--h33-script",required=True)
 p.add_argument("--panel",required=True); p.add_argument("--vep-results",required=True); p.add_argument("--g55-results",required=True); p.add_argument("--h22-results",required=True)
 p.add_argument("--device",default="cuda"); p.add_argument("--test-trajectories",type=int,default=20000); p.add_argument("--jobs",type=int,default=4); p.add_argument("--equivalence-check",action="store_true"); a=p.parse_args()
 if js(a.protocol).get("design_sha256")!=HASH: raise SystemExit("protocol hash mismatch")
 if a.jobs<1 or a.jobs>8: raise SystemExit("--jobs must be in 1..8; start with 2 on RTX 3060 Laptop")
 out=Path(a.results); out.mkdir(parents=True,exist_ok=True); d3,h,raw,*rest=setup(a); ctx=(h,*rest)
 if a.stage=="train": train(a,d3,raw,ctx,out)
 elif a.stage=="validation": validation(a,d3,raw,ctx,out)
 else:
  if a.test_trajectories<20000: raise SystemExit("TEST N<20000 forbidden")
  test(a,d3,raw,ctx,out)
if __name__=="__main__": main()
