#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RESS V1-I.5-D.3.2-FAST. Diagnostic only; NO NEW SIMULATION."""
import argparse,json,math,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar

def prep(C):
    S=np.sort(np.asarray(C,float),axis=1)
    CS=np.cumsum(S[:,::-1],axis=1)[:,::-1]
    return S,np.c_[CS,np.zeros(len(S))]

def geta(S,CS,eta,idx):
    X=S[idx]; Y=CS[idx]; L=X.shape[1]
    k=np.sum(X<=eta,axis=1); n=L-k
    return (Y[np.arange(len(idx)),k]-n*eta)/L

def klsup(G,eps):
    M=len(G)
    def q(lam):
        z=G/lam; z-=z.max(); w=np.exp(z); return w/w.sum()
    def kl(lam):
        w=q(lam); return float(np.sum(w*np.log(np.maximum(w*M,1e-300))))
    lo=1e-14; hi=max(float(np.std(G)),float(np.max(np.abs(G))),1e-10)
    while kl(hi)>eps: hi*=2
    for _ in range(70):
        mid=math.sqrt(lo*hi)
        if kl(mid)>eps: lo=mid
        else: hi=mid
    w=q(hi); return float(w@G),w,float(hi),kl(hi)

def solve(S,CS,idx,alpha=.95,eps=.08):
    idx=np.asarray(idx,int); lo=float(S[idx].min()); hi=float(S[idx].max())
    def f(x):
        G=geta(S,CS,float(x),idx); sup,_,_,_=klsup(G,eps)
        return float(x+sup/(1-alpha))
    o=minimize_scalar(f,bounds=(lo,hi),method="bounded",
        options={"xatol":max(1e-13,(hi-lo)*1e-10),"maxiter":100})
    eta=float(o.x); G=geta(S,CS,eta,idx); sup,q,lam,kl=klsup(G,eps)
    return dict(rho=float(eta+sup/(1-alpha)),eta=eta,G=G,q=q,lam=lam,kl=kl)

def summary(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),
      median=float(np.median(x)),q025=float(np.quantile(x,.025)),
      q975=float(np.quantile(x,.975)),min=float(x.min()),max=float(x.max()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--matrix",default="Results_RESS_V1I5D31/tables/v1i5d31_BTCUSDT_cost_matrix_M512_L2048.npy")
    p.add_argument("--atoms",default="Results_RESS_V1I5D31/tables/v1i5d31_BTCUSDT_outer_atoms_M512.csv")
    p.add_argument("--results",default="Results_RESS_V1I5D32_FAST")
    p.add_argument("--permutations",type=int,default=40); p.add_argument("--seed",type=int,default=101)
    a=p.parse_args(); t=time.time()
    C=np.load(a.matrix); A=pd.read_csv(a.atoms)
    if C.shape!=(512,2048) or not np.isfinite(C).all(): raise RuntimeError("Need finite 512x2048 matrix")
    if len(A)!=512: raise RuntimeError("Need 512 atoms")
    out=Path(a.results); (out/"tables").mkdir(parents=True,exist_ok=True)
    print("INPUT PASS | 512x2048 | NO NEW SIMULATION",flush=True)
    S,CS=prep(C); print("PRECOMPUTE PASS",flush=True)

    pref=[]; prev=None
    for M in (128,256,384,512):
        r=solve(S,CS,np.arange(M)); z=dict(M=M,rho=r["rho"],eta=r["eta"],
          neff=float(1/np.sum(r["q"]**2)),qmax=float(r["q"].max()),achieved_kl=r["kl"])
        if prev is not None: z["relative_change_vs_previous"]=abs(z["rho"]-prev)/max(abs(z["rho"]),1e-18)
        pref.append(z); prev=z["rho"]; print(f"PREFIX M={M} | rho={z['rho']:.12g}",flush=True)
    pd.DataFrame(pref).to_csv(out/"tables"/"prefixes.csv",index=False)
    print(f"ORIGINAL 384->512 GATE = {pref[-1]['relative_change_vs_previous']:.4%} | FAIL RETAINED",flush=True)

    rng=np.random.default_rng(np.random.SeedSequence([a.seed,320001])); rows=[]
    for b in range(a.permutations):
        perm=rng.permutation(512)
        for M in (128,256,384):
            r=solve(S,CS,perm[:M]); rows.append(dict(rep=b,M=M,rho=r["rho"],eta=r["eta"],
              neff=float(1/np.sum(r["q"]**2)),qmax=float(r["q"].max())))
        print(f"PERM {b+1}/{a.permutations}",flush=True)
    R=pd.DataFrame(rows); R.to_csv(out/"tables"/"permutations.csv",index=False)

    blocks=[]
    for M in (128,256):
        for b,s in enumerate(range(0,512,M)):
            r=solve(S,CS,np.arange(s,s+M)); blocks.append(dict(M=M,block=b,rho=r["rho"],eta=r["eta"]))
    B=pd.DataFrame(blocks); B.to_csv(out/"tables"/"blocks.csv",index=False)
    print("BLOCKS PASS",flush=True)

    full=solve(S,CS,np.arange(512)); q,G=full["q"],full["G"]; qG=q*G
    screen=pd.DataFrame(dict(outer_id=np.arange(512),q=q,G=G,qG=qG,
      contribution_share=np.abs(qG)/max(abs(qG.sum()),1e-300))).sort_values("contribution_share",ascending=False)
    screen.to_csv(out/"tables"/"influence_screen.csv",index=False)

    # Exact FAST LOO only for top 20 screened atoms.
    loo=[]; allidx=np.arange(512)
    for j,i in enumerate(screen.outer_id.head(20).astype(int)):
        r=solve(S,CS,allidx[allidx!=i])
        loo.append(dict(outer_id=i,rho_loo=r["rho"],
          relative_influence=abs(r["rho"]-full["rho"])/max(abs(full["rho"]),1e-18)))
        print(f"FAST LOO {j+1}/20 | outer={i}",flush=True)
    LOO=pd.DataFrame(loo).sort_values("relative_influence",ascending=False)
    LOO.to_csv(out/"tables"/"exact_loo_top20.csv",index=False)

    ps={str(M):summary(R.loc[R.M==M,"rho"]) for M in (128,256,384)}
    bs={str(M):summary(B.loc[B.M==M,"rho"]) for M in (128,256)}
    ans=dict(version="V1-I.5-D.3.2-FAST",no_new_simulation=True,
      original_gate_status="FAIL retained; 5% rule unchanged",original_prefixes=pref,
      permutation_prefix_summary=ps,disjoint_block_summary=bs,
      full_M512=dict(rho=full["rho"],eta=full["eta"],neff=float(1/np.sum(q*q)),
        qmax=float(q.max()),top_outer_id=int(np.argmax(qG)),top1_qG_share=float(qG.max()/qG.sum())),
      exact_loo_top20=dict(max_relative_influence=float(LOO.relative_influence.max()),
        median_relative_influence=float(LOO.relative_influence.median()),
        most_influential_outer_id=int(LOO.iloc[0].outer_id)),
      guards=dict(train_only=True,no_validation=True,no_test=True,no_policy_selection=True,
        no_radius_selection=True,original_5pct_gate_not_redefined=True),
      elapsed_seconds=time.time()-t)
    (out/"V1I5D32_FAST_SUMMARY.json").write_text(json.dumps(ans,indent=2),encoding="utf-8")
    print(json.dumps(ans,indent=2)); print(f"DONE | elapsed={time.time()-t:.1f}s | {out.resolve()}")

if __name__=="__main__": main()
