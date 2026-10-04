#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RESS V1-H.1 — Coupled sequential execution environment:
locked G.5.5-style exogenous market recursion + execution accounting + ex-ante stress grid.

PURPOSE
-------
Development audit before RL/DRO. The action consumes contemporaneous modeled
bid-side liquidity and generates execution cost, but DOES NOT feed back into
the exogenous market transition. No causal market-impact claim is made.

Important:
* theta is sampled internally and is NEVER a policy input;
* all policies use observable state/history only;
* the same policy rule is used across latent scenarios;
* no RL and no DRO are fitted here;
* inventory/horizon stress grid is fixed ex ante;
* exact multi-level historical book reconstruction is NOT claimed when the
  upstream state contains only aggregate K-level depth.
"""

import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd

XI=["dlog_mid","dlog_total_depth","d_log_depth_ratio"]
XFULL=["lag_dlog_mid","lag_dlog_total_depth","lag_d_log_depth_ratio",
       "lag_log_spread","lag_imbalance","lag_log_rv"]
EPS=1e-12

def find_panel(p):
    p=Path(p)
    if p.is_file(): return p
    xs=sorted(p.glob("*.parquet"))
    if not xs: raise FileNotFoundError(f"No parquet panel under {p}")
    pref=[x for x in xs if "causal_grid_panel" in x.name]
    return pref[0] if pref else xs[0]

def load_panel(p):
    q=find_panel(p); return pd.read_parquet(q),q

def slog(x):
    z=pd.to_numeric(x,errors="coerce")
    return np.log(z.where(z>0))

def build_raw(d,grid,K):
    d=d[pd.to_numeric(d.grid_seconds,errors="coerce").eq(grid)].copy()
    if "K" in d:
        d=d[pd.to_numeric(d.K,errors="coerce").eq(K)].copy()
    d=d.sort_values(["symbol","date","grid_timestamp"]).reset_index(drop=True)
    d["mid"]=pd.to_numeric(d.mid,errors="coerce")
    d["bid_depth"]=pd.to_numeric(d.bid_depth,errors="coerce")
    d["ask_depth"]=pd.to_numeric(d.ask_depth,errors="coerce")
    d["spread_bps"]=pd.to_numeric(d.spread_bps,errors="coerce")
    d["rv_backward"]=pd.to_numeric(d.rv_backward,errors="coerce")
    d["total_depth"]=d.bid_depth+d.ask_depth
    d["log_mid"]=slog(d.mid)
    d["log_total_depth"]=slog(d.total_depth)
    d["log_depth_ratio"]=slog(d.bid_depth)-slog(d.ask_depth)
    G=d.groupby(["symbol","date"],sort=False)
    d["dlog_mid"]=G.log_mid.diff()
    d["dlog_total_depth"]=G.log_total_depth.diff()
    d["d_log_depth_ratio"]=G.log_depth_ratio.diff()
    for v in XI: d["lag_"+v]=G[v].shift(1)
    d["log_spread_now"]=slog(d.spread_bps)
    d["log_rv_now"]=slog(d.rv_backward)
    d["lag_log_spread"]=slog(G.spread_bps.shift(1))
    d["lag_imbalance"]=G.imbalance.shift(1)
    d["lag_log_rv"]=slog(G.rv_backward.shift(1))
    d["next_log_spread"]=slog(d.spread_bps)
    return d

def load_json(path):
    with open(path,"r",encoding="utf8") as f:return json.load(f)

def load_npz_resid(path):
    z=np.load(path,allow_pickle=False)
    return {k:np.asarray(z[k],float) for k in z.files}

def reconstruct_depth(total,ratio):
    total=max(float(total),EPS); r=float(ratio)
    if r>=0:
        db=total/(1.0+math.exp(-min(r,745.0)))
    else:
        er=math.exp(max(r,-745.0)); db=total*er/(1.0+er)
    da=max(total-db,EPS); db=max(db,EPS)
    imb=(db-da)/(db+da)
    return db,da,imb

def inv_std(name,z,sc):
    d=sc[name]
    sd=float(d["sd"]) if np.isfinite(float(d["sd"])) and float(d["sd"])>0 else 1.0
    return float(d["mean"])+sd*float(z)

def center_std(symbol,xstd,center_fits):
    B=np.asarray(center_fits[symbol],float)
    return np.r_[1.,np.asarray(xstd,float)]@B

def make_model_classes(torch,nn,F):
    class Posterior(nn.Module):
        def __init__(self,din,h):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(din,h),nn.Tanh(),
                                   nn.Linear(h,h),nn.Tanh())
            self.mu=nn.Linear(h,1); self.logsd=nn.Linear(h,1)
        def forward(self,x):
            h=self.net(x)
            return self.mu(h).squeeze(-1),torch.clamp(self.logsd(h).squeeze(-1),-5,2)

    class PriorScale(nn.Module):
        def __init__(self,din,h,conditional,mu_bound,sd_min,sd_max):
            super().__init__()
            self.conditional=conditional; self.mu_bound=float(mu_bound)
            self.sd_min=float(sd_min); self.sd_max=float(sd_max)
            if conditional:
                self.net=nn.Sequential(nn.Linear(din,h),nn.Tanh(),
                                       nn.Linear(h,h),nn.Tanh())
                self.mu_head=nn.Linear(h,1); self.sd_head=nn.Linear(h,1)
            else:
                self.raw_mu=nn.Parameter(torch.tensor(0.0))
                self.raw_sd=nn.Parameter(torch.tensor(-0.5))
        def raw_moments(self,x):
            if self.conditional:
                h=self.net(x)
                mu=self.mu_bound*torch.tanh(self.mu_head(h).squeeze(-1))
                sd=self.sd_min+(self.sd_max-self.sd_min)*torch.sigmoid(
                    self.sd_head(h).squeeze(-1))
            else:
                mu=self.mu_bound*torch.tanh(self.raw_mu).expand(x.shape[0])
                sd=(self.sd_min+(self.sd_max-self.sd_min)*
                    torch.sigmoid(self.raw_sd)).expand(x.shape[0])
            return mu,sd
        def moments(self,x,log_norm=0.0):
            mu,sd=self.raw_moments(x)
            return mu-log_norm,sd

    class PredictiveLatent(nn.Module):
        def __init__(self,xdim,h,conditional,mu_bound,sd_min,sd_max):
            super().__init__()
            self.posterior=Posterior(xdim+3,h)
            self.prior=PriorScale(xdim,h,conditional,mu_bound,sd_min,sd_max)
            self.raw_nu=nn.Parameter(torch.tensor(0.5))
            self.Lraw=nn.Parameter(torch.eye(3))
        def noise(self):
            nu=2.05+F.softplus(self.raw_nu)
            L=torch.tril(self.Lraw)
            dg=F.softplus(torch.diagonal(L))+1e-4
            L=L-torch.diag(torch.diagonal(L))+torch.diag(dg)
            S=L@L.T; s=torch.sqrt(torch.diagonal(S))
            R=S/(s[:,None]*s[None,:])
            return nu,R
    return PredictiveLatent

def load_frozen_m2(vep_results,device,torch,nn,F):
    root=Path(vep_results)
    ck=root/"models"/"v1f53_models.pt"
    if not ck.exists(): raise FileNotFoundError(f"Missing frozen checkpoint: {ck}")
    obj=torch.load(ck,map_location=device,weights_only=False)
    cfg=obj["config"]; xcols=obj["model_predictors"]["M2_predictive_full"]
    Model=make_model_classes(torch,nn,F)
    m=Model(len(xcols),int(cfg["hidden"]),True,float(cfg["mu_bound"]),
            float(cfg["prior_sd_min"]),float(cfg["prior_sd_max"])).to(device)

    # Strict architectural compatibility with the frozen V1-F.5.3 model.
    # Never use strict=False here: a silently uninitialized dependence matrix
    # would invalidate the simulated multivariate innovations.
    frozen_state=obj["models"]["M2_predictive_full"]
    expected=set(m.state_dict().keys())
    received=set(frozen_state.keys())
    missing=sorted(expected-received)
    unexpected=sorted(received-expected)
    if missing or unexpected:
        raise RuntimeError(
            "V1-F.5.3 checkpoint architecture mismatch. "
            f"Missing keys={missing}; unexpected keys={unexpected}"
        )
    m.load_state_dict(frozen_state,strict=True)
    m.eval()
    ln=float(obj["train_global_log_normalizers"]["M2_predictive_full"])
    return m,xcols,ln,obj["scaler"],obj["center_fits"],cfg

def sample_m2(model,x,log_norm,rng,torch,device,noise_cache):
    """Frozen F.5.3 M2 draw; cached frozen noise law; no theta clipping."""
    nu,R=noise_cache
    with torch.inference_mode():
        xt=torch.as_tensor(np.asarray(x,np.float32)[None,:],device=device)
        mu,sd=model.prior.moments(xt,log_norm)
        mu=float(mu.detach().cpu()[0]); sd=float(sd.detach().cpu()[0])
    z=rng.normal(mu,sd)
    theta=math.exp(z)
    g=rng.multivariate_normal(np.zeros(3),R)
    u=rng.chisquare(nu)
    eps=g/math.sqrt(u/nu)
    return math.sqrt(theta)*eps,theta,mu,sd

def liq_center(spec,y,z):
    A=np.asarray(spec["A"],float); B=np.asarray(spec["B"],float)
    zm=np.asarray(spec["exog_mean"],float); zs=np.asarray(spec["exog_sd"],float)
    zz=(np.asarray(z,float)-zm)/zs
    return np.asarray(spec["intercept"],float)+A@np.asarray(y,float)+B@zz

def spread_next(spec,spread_bps,feat,rng,resid):
    x=np.array([feat[c] for c in spec["exog_names"]],float)
    z=(x-np.asarray(spec["exog_mean"],float))/np.asarray(spec["exog_sd"],float)
    c=float(spec["intercept"])+float(spec["phi"])*math.log(spread_bps)
    c+=float(np.dot(np.asarray(spec["gamma"],float),z))
    ln=c+float(rng.choice(resid))
    return math.exp(ln)  # expose overflow naturally

def load_empirical_book(h22_results,K):
    root=Path(h22_results)
    shp=root/"models"/"v1h22_train_joint_book_shapes.parquet"
    edg=root/"models"/"v1h22_symbol_train_bin_edges.csv"
    mp=root/"models"/"v1h22_sampling_map.csv"
    for p in [shp,edg,mp]:
        if not p.exists(): raise FileNotFoundError(p)
    shapes=pd.read_parquet(shp)
    edges=pd.read_csv(edg)
    smap=pd.read_csv(mp)
    need=["symbol","spread_bin","depth_bin","imb_bin"]+[f"w{k}" for k in range(1,K+1)]+[f"delta{k}" for k in range(1,K+1)]
    miss=[c for c in need if c not in shapes]
    if miss: raise KeyError(f"H.2.2 shapes missing columns: {miss}")
    edge_map={}
    # H.2.2 stores internal raw-variable labels in the edge file.
    # Normalize them once at load time to the semantic names used by H.3.
    edge_name_map={
        "_spread_x":"spread_bps",
        "_total_depth":"total_depth",
        "_imb_x":"imbalance",
        "spread_bps":"spread_bps",
        "total_depth":"total_depth",
        "imbalance":"imbalance",
    }
    for (sym,var),g in edges.groupby(["symbol","variable"]):
        var=str(var)
        if var not in edge_name_map:
            raise KeyError(f"Unknown H.2.2 edge variable: {var}")
        semantic=edge_name_map[var]
        edge_map[(str(sym),semantic)]=g.sort_values("edge_index")["edge"].to_numpy(float)
    for sym in shapes["symbol"].astype(str).unique():
        for semantic in ("spread_bps","total_depth","imbalance"):
            if (sym,semantic) not in edge_map:
                raise KeyError(f"Missing normalized H.2.2 TRAIN edges: {(sym,semantic)}")
    pools={}
    symbol_pools={}
    for sym,g in shapes.groupby("symbol"):
        sym=str(sym); symbol_pools[sym]=g.index.to_numpy(int)
        for key,q in g.groupby(["spread_bin","depth_bin","imb_bin"]):
            pools[(sym,)+tuple(int(x) for x in key)]=q.index.to_numpy(int)
    eligible=set()
    for _,r in smap.iterrows():
        if bool(r["eligible_cell"]):
            eligible.add((str(r["symbol"]),int(r["spread_bin"]),int(r["depth_bin"]),int(r["imb_bin"])))
    return shapes,edge_map,pools,symbol_pools,eligible

def frozen_bin(x,edges):
    # TRAIN edges already have -inf/+inf endpoints.
    j=int(np.searchsorted(np.asarray(edges,float),float(x),side="right")-1)
    return max(0,min(j,len(edges)-2))

def sample_book_shape(sym,spread_bps,total_depth,imb,book,rng,K):
    shapes,edges,pools,symbol_pools,eligible=book
    vals=[("spread_bps",spread_bps),("total_depth",total_depth),("imbalance",imb)]
    bins=[]
    for var,x in vals:
        key=(str(sym),var)
        if key not in edges: raise KeyError(f"Missing H.2.2 TRAIN edges: {key}")
        bins.append(frozen_bin(x,edges[key]))
    cell=(str(sym),bins[0],bins[1],bins[2])
    exact=cell in eligible and cell in pools and len(pools[cell])>0
    idxs=pools[cell] if exact else symbol_pools.get(str(sym))
    if idxs is None or len(idxs)==0: raise RuntimeError(f"No H.2.2 TRAIN shape pool for {sym}")
    row=shapes.loc[int(rng.choice(idxs))]
    w=np.array([float(row[f"w{k}"]) for k in range(1,K+1)],float)
    delta=np.array([float(row[f"delta{k}"]) for k in range(1,K+1)],float)
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(delta)): raise RuntimeError("Nonfinite empirical book shape.")
    if np.any(w<0) or abs(w.sum()-1)>1e-8: raise RuntimeError("Invalid empirical book weights.")
    if np.any(np.diff(delta)<-1e-10): raise RuntimeError("Invalid empirical bid-price geometry.")
    return w,delta,exact,tuple(bins)

def walk_bid(mid,spread,bid_depth,qty,shares,mults):
    qty=max(float(qty),0.0); rem=qty; fill=0.; notional=0.
    for cap,m in zip(max(bid_depth,0.)*shares,mults):
        take=min(rem,float(cap))
        px=max(mid-m*spread,EPS)
        fill+=take; notional+=take*px; rem-=take
        if rem<=EPS: break
    if fill<=0:return 0.,np.nan,0.,qty
    vwap=notional/fill
    return fill,vwap,fill*(mid-vwap),max(rem,0.)


def forced_terminal_walk(mid,spread,bid_depth,qty,shares,mults):
    """Mechanical terminal liquidation through the observed/simulated K-level bid book.
    No extrapolation beyond K: any residual is reported as unliquidated."""
    qty=max(float(qty),0.0)
    if qty<=0:
        return 0.0,np.nan,0.0,0.0
    fill,vwap,cost,_=walk_bid(mid,spread,bid_depth,qty,shares,mults)
    residual=max(qty-fill,0.0)
    return fill,vwap,cost,residual

def action_rule(policy,q,steps_left,bid_depth,pov,max_participation):
    if q<=0:return 0.
    if policy=="twap": target=q/max(steps_left,1)
    elif policy=="pov": target=pov*bid_depth
    elif policy=="front_loaded": target=1.5*q/max(steps_left,1)
    else: raise ValueError(policy)
    # Common admissible action set for every policy.
    cap=max(float(max_participation),0.0)*max(float(bid_depth),0.0)
    return min(float(q),max(float(target),0.0),cap)

def prepare_initial(raw,horizon):
    d=raw[raw.split.eq("test")].copy() if "split" in raw else raw.copy()
    need=["symbol","mid","total_depth","log_depth_ratio","spread_bps","imbalance",
          "rv_backward"]+XI
    return d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)

def simulate_episode(r0,sym,H,inv_frac,policy,pov,max_participation,K,model,ln,sc,center_fits,
                     liq,spread_spec,spread_resid,book,rng,torch,device,rv_window,noise_cache,
                     max_terminal_steps):
    mid=float(r0.mid); total=float(r0.total_depth); ratio=float(r0.log_depth_ratio)
    spread_bps=float(r0.spread_bps); imb=float(r0.imbalance)
    rv=max(float(r0.rv_backward),EPS)
    last=np.array([float(r0[c]) for c in XI])
    db,_,_=reconstruct_depth(total,ratio)
    q0=max(inv_frac*db,EPS); q=q0
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
        a=action_rule(policy,q,H-t,db,pov,max_participation)
        fill,vwap,cost,_=walk_bid(mid,spread,db,a,shares,mults)
        q-=fill; filled_total+=fill; sf+=cost; regular_sf+=cost
        if db>0:max_part=max(max_part,fill/db)

        xraw=np.array([last[0],last[1],last[2],math.log(spread_bps),imb,math.log(rv)])
        xstd=np.array([(xraw[j]-float(sc[XFULL[j]]["mean"]))/
                       max(float(sc[XFULL[j]]["sd"]),EPS) for j in range(len(XFULL))])
        e,theta,_,_=sample_m2(model,xstd,ln,rng,torch,device,noise_cache)
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
        e,theta,_,_=sample_m2(model,xstd,ln,rng,torch,device,noise_cache)
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

def cvar(x,a):
    """Exact empirical Rockafellar-Uryasev envelope at the empirical VaR minimizer."""
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x):return np.nan
    if not 0<a<1: raise ValueError("alpha must lie in (0,1)")
    eta=float(np.quantile(x,a,method="inverted_cdf"))
    return float(eta+np.mean(np.maximum(x-eta,0.0))/(1.0-a))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--panel",default="Results_RESS_V1D/panels")
    ap.add_argument("--vep-results",default="Results_RESS_V1F53")
    ap.add_argument("--g55-results",default="Results_RESS_V1G55")
    ap.add_argument("--h22-results",default="Results_RESS_V1H22")
    ap.add_argument("--results",default="Results_RESS_V1H33")
    ap.add_argument("--grid-seconds",type=int,default=5)
    ap.add_argument("--K",type=int,default=10)
    ap.add_argument("--episodes",type=int,default=2000,
                    help="episodes per symbol x stress cell x policy")
    ap.add_argument("--inventory-grid",default="0.2,0.5,1.0,2.0")
    ap.add_argument("--horizon-grid",default="30,60,120")
    ap.add_argument("--participation",type=float,default=.10,
                    help="POV target fraction of contemporaneous bid depth")
    ap.add_argument("--max-participation",type=float,default=.10,
                    help="COMMON per-step admissible cap as fraction of contemporaneous bid depth")
    ap.add_argument("--max-terminal-steps",type=int,default=120,
                    help="Maximum mechanical settlement steps after H; audit parameter, not outcome-tuned")
    ap.add_argument("--alpha",type=float,default=.95)
    ap.add_argument("--rv-window",type=int,default=12)
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()
    if a.max_terminal_steps<1: raise ValueError("max-terminal-steps must be >=1")
    if not (0.0<a.max_participation<=1.0): raise ValueError("max-participation must lie in (0,1]")
    invs=[float(x) for x in a.inventory_grid.split(",")]
    horizons=[int(x) for x in a.horizon_grid.split(",")]
    if any(x<=0 for x in invs) or any(h<=0 for h in horizons):raise ValueError("Positive grids required.")
    out=Path(a.results)
    for q in ["tables","figure_sources"]: (out/q).mkdir(parents=True,exist_ok=True)

    raw,panel_path=load_panel(a.panel)
    raw=build_raw(raw,a.grid_seconds,a.K)
    # All NumPy/MKL preprocessing before lazy torch import.
    liq=load_json(Path(a.g55_results)/"models"/"v1g55_liquidity_varx.json")
    spr=load_json(Path(a.g55_results)/"models"/"v1g55_spread_arx.json")
    sres=load_npz_resid(Path(a.g55_results)/"models"/"v1g55_spread_residuals.npz")
    for sym,p in liq.items():
        if not p["stable_train"]:raise RuntimeError(f"Unstable liquidity kernel: {sym}")
    for sym,p in spr.items():
        if not p["stable_train"]:raise RuntimeError(f"Unstable spread kernel: {sym}")
    init=prepare_initial(raw,max(horizons))
    book=load_empirical_book(a.h22_results,a.K)
    print(f"EMPIRICAL BOOK PASS | TRAIN shapes={len(book[0]):,} | eligible cells={len(book[4])}",flush=True)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    model,xcols,ln,sc,center_fits,cfg=load_frozen_m2(a.vep_results,device,torch,nn,F)
    if xcols!=XFULL:raise RuntimeError(f"Frozen XFULL mismatch: {xcols}")
    with torch.inference_mode():
        nu_t,R_t=model.noise()
        noise_cache=(float(nu_t.detach().cpu()),R_t.detach().cpu().numpy())
    print(f"FROZEN MODEL PASS | device={device} | nu={noise_cache[0]:.4f}",flush=True)

    rng=np.random.default_rng(a.seed)
    policies=["twap","pov","front_loaded"]
    rows=[]
    # Common initial-state indices within each symbol/stress cell across policies.
    for sym in sorted(init.symbol.unique()):
        ds=init[init.symbol.eq(sym)].reset_index(drop=True)
        if not len(ds):continue
        for H in horizons:
            for inv in invs:
                idx=rng.integers(0,len(ds),size=a.episodes)
                for eid,j in enumerate(idx):
                    r0=ds.iloc[j]
                    # Common random-number seed per episode; policy does not alter market law.
                    base_seed=int(a.seed+1000003*eid+1009*H+int(inv*10000)+
                                  sum(ord(c) for c in sym))
                    for pol in policies:
                        prng=np.random.default_rng(base_seed)
                        z=simulate_episode(r0,sym,H,inv,pol,a.participation,a.max_participation,a.K,
                                           model,ln,sc,center_fits,liq,spr,sres,book,
                                           prng,torch,device,a.rv_window,noise_cache,
                                           a.max_terminal_steps)
                        z.update({"symbol":sym,"horizon":H,"inventory_fraction":inv,
                                  "policy":pol,"episode":eid})
                        rows.append(z)
                    if (eid+1)==1 or (eid+1)%100==0 or (eid+1)==a.episodes:
                        print(f"PROGRESS | {sym} | H={H} | inv={inv:g} | "
                              f"{eid+1}/{a.episodes} episodes",flush=True)
    res=pd.DataFrame(rows)
    res.to_csv(out/"figure_sources"/"v1h33_coupled_episode_results.csv.gz",
               index=False,compression="gzip")
    sm=[]
    for keys,d in res.groupby(["symbol","horizon","inventory_fraction","policy"]):
        sym,H,inv,pol=keys; x=d.shortfall_bps.to_numpy(float)
        sm.append({"symbol":sym,"horizon":H,"inventory_fraction":inv,"policy":pol,
                   "n":len(d),"mean_shortfall_bps":float(np.nanmean(x)),
                   "p95_shortfall_bps":float(np.nanquantile(x,.95)),
                   f"cvar_{a.alpha:.2f}_shortfall_bps":cvar(x,a.alpha),
                   "mean_completion_rate":float(d.completion_rate.mean()),
                   "failure_rate_terminal_inventory":float((d.terminal_inventory>1e-10).mean()),
                   "numerical_failure_rate":float(d.numerical_failure.mean()),
                    "mean_max_step_participation":float(d.max_step_participation.mean()),
                   "mean_book_fallback_step_rate":float(d.book_fallback_step_rate.mean()),
                   "terminal_liquidation_frequency":float(d.terminal_liquidation_required.mean()),
                   "mean_terminal_settlement_steps":float(d.terminal_settlement_steps.mean()),
                   "p95_terminal_settlement_steps":float(d.terminal_settlement_steps.quantile(.95)),
                   "max_terminal_settlement_steps_used":int(d.terminal_settlement_steps.max()),
                   "terminal_settlement_exhaustion_rate":float(d.terminal_settlement_exhausted.mean()),
                   "terminal_settlement_numerical_failure_rate":float(d.terminal_settlement_numerical_failure.mean()),
                   "mean_pre_terminal_inventory_fraction":float((d.pre_terminal_inventory/d.q0).mean()),
                   "mean_terminal_unliquidated_fraction":float(d.terminal_unliquidated_fraction.mean()),
                   "mean_terminal_forced_shortfall_bps":float(d.terminal_forced_shortfall_bps.mean()),
                   "terminal_book_fallback_step_rate":float(d.terminal_book_fallback_step_rate.mean())})
    pd.DataFrame(sm).to_csv(out/"tables"/"v1h33_stress_grid_summary.csv",index=False)
    meta={"version":"V1-H.3.3","role":"empirical-K10 coupled development sequential execution environment",
          "input_panel":str(panel_path.resolve()),"grid_seconds":a.grid_seconds,"g55_results":str(Path(a.g55_results).resolve()),
          "inventory_grid":invs,"horizon_grid":horizons,"episodes_per_cell_policy":a.episodes,
          "policies":policies,"theta_policy_input":False,
          "theta_fields_in_output_are_ex_post_diagnostics_only":True,
          "market_path_feedback_from_action":False,"causal_market_impact_identified":False,
           "book_model":"H.2.2 TRAIN-calibrated joint K-level bid shapes conditioned on observable simulated liquidity bins",
          "h22_results":str(Path(a.h22_results).resolve()),
          "common_action_set":"0 <= a_t <= min(q_t, max_participation * contemporaneous K-level bid depth)",
          "max_participation":a.max_participation,
           "empirical_cvar":"finite-sample Rockafellar-Uryasev envelope",
          "terminal_rule":"multi-step mechanical settlement through successive empirical K-level bid books with frozen exogenous G.5.5 transitions between books; no extrapolation beyond K",
          "terminal_participation_cap":"not applied to forced terminal settlement",
          "terminal_insufficient_depth":"residual inventory carries to the next settlement step; no invented K+ levels",
          "max_terminal_steps":a.max_terminal_steps,
          "terminal_policy_decisions":False,
          "common_random_numbers_across_policies":True,
          "rl_fitted":False,"dro_fitted":False,
          "purpose":"multi-step terminal-settlement closure audit before policy optimization",
          "development_panel_warning":"First-of-month Tardis sample is not publication-representative."}
    with open(out/"V1H1_SUMMARY.json","w",encoding="utf8") as f:json.dump(meta,f,indent=2)
    print("DONE | V1-H.3.3 empirical-K10 coupled sequential execution audit")
    print("Results ->",out.resolve())

if __name__=="__main__":
    main()
