#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RESS V1-I.7 — Final Statistical Analysis & Publication Package

Post-TEST analysis only. This script NEVER simulates trajectories and NEVER
retunes policies. It consumes the locked V1-I.6-TURBO outputs and produces:
  A) integrity/reproducibility audit
  B) paired seed-level policy inference
  C) reliability decomposition
  D) publication-ready CSV/LaTeX/figures + final JSON summary

Primary comparison: localized_kl versus each of
expected_cost, nominal_cvar, global_kl.

Inference unit: seed (15 predeclared seeds), preserving the paired CRN design.
Bootstrap: paired nonparametric bootstrap over seeds.
Secondary tests: exact two-sided sign test and Wilcoxon signed-rank when SciPy
is available. These are diagnostics; bootstrap CIs and effect sizes are primary.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "V1-I.7"
EXPECTED_OBJECTIVES = ("expected_cost", "nominal_cvar", "global_kl", "localized_kl")
EXPECTED_ASSETS = ("BTCUSDT", "ETHUSDT")
EXPECTED_SEEDS = (101,202,303,404,505,606,707,808,909,1001,1111,1212,1313,1414,1515)
PRIMARY = "localized_kl"
COMPARATORS = ("expected_cost", "nominal_cvar", "global_kl")
METRICS = (
    "mean", "cvar95", "cvar99", "exceedance_rate",
    "mean_excess_given_exceedance", "completion"
)
LOWER_IS_BETTER = {
    "mean": True, "cvar95": True, "cvar99": True,
    "exceedance_rate": True, "mean_excess_given_exceedance": True,
    "completion": False,
}
BOOTSTRAP_REPS = 50000
BOOTSTRAP_SEED = 710001
TOL = 1e-12


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_series(s):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return bool(np.all(np.isfinite(x)))


def exact_sign_p(nneg, npos):
    """Exact two-sided binomial sign test, dropping ties."""
    n = int(nneg + npos)
    if n == 0:
        return 1.0
    k = min(int(nneg), int(npos))
    tail = sum(math.comb(n, j) for j in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap(delta, reps, rng):
    delta = np.asarray(delta, float)
    n = len(delta)
    # Chunked to keep memory tiny and deterministic.
    out = np.empty(reps, float)
    chunk = 5000
    p = 0
    while p < reps:
        b = min(chunk, reps - p)
        idx = rng.integers(0, n, size=(b, n))
        out[p:p+b] = delta[idx].mean(axis=1)
        p += b
    return (
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
    )


def safe_pct(delta, comparator):
    if abs(comparator) <= 1e-15:
        return np.nan
    return 100.0 * delta / comparator


def latex_escape(s):
    return str(s).replace("_", r"\_")


def write_latex_table(df, path, caption, label, float_format="%.6g"):
    # pandas emits a self-contained tabular. Wrap it explicitly for publication use.
    body = df.to_latex(index=False, escape=True, float_format=float_format)
    txt = (
        "\\begin{table}[!htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        + body +
        "\\end{table}\n"
    )
    Path(path).write_text(txt, encoding="utf-8")


def audit_inputs(root, T, B, complete, train_lock, val_lock, manifest):
    checks = []
    def add(name, passed, detail):
        checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})

    add("TEST_COMPLETE.complete", complete.get("complete") is True, complete.get("complete"))
    add("TEST_COMPLETE.test_once", complete.get("test_once") is True, complete.get("test_once"))
    add("TEST sample size lock", int(complete.get("n_per_policy_seed", -1)) >= 20000,
        complete.get("n_per_policy_seed"))
    add("TRAIN locked", train_lock.get("locked") is True, train_lock.get("locked"))
    add("VALIDATION no retuning", val_lock.get("confirmed_without_retuning") is True,
        val_lock.get("confirmed_without_retuning"))
    add("Manifest common random numbers", manifest.get("common_random_numbers") is True,
        manifest.get("common_random_numbers"))

    required = {
        "asset","seed","objective","gamma","n","mean","cvar95","cvar99",
        "train_frozen_exceedance_threshold","exceedance_rate",
        "mean_excess_given_exceedance","completion"
    }
    add("TEST_SEED_METRICS required columns", required.issubset(T.columns),
        sorted(required - set(T.columns)) if not required.issubset(T.columns) else "all present")

    expected_rows = len(EXPECTED_ASSETS)*len(EXPECTED_SEEDS)*len(EXPECTED_OBJECTIVES)
    add("TEST row count", len(T) == expected_rows, f"{len(T)} / {expected_rows}")
    add("No duplicate asset-seed-objective",
        not T.duplicated(["asset","seed","objective"]).any(),
        int(T.duplicated(["asset","seed","objective"]).sum()))
    add("Expected assets", set(T["asset"]) == set(EXPECTED_ASSETS), sorted(set(T["asset"])))
    add("Expected objectives", set(T["objective"]) == set(EXPECTED_OBJECTIVES),
        sorted(set(T["objective"])))
    add("Expected seeds", set(map(int,T["seed"])) == set(EXPECTED_SEEDS),
        sorted(set(map(int,T["seed"]))))

    numeric_cols = ["gamma","n","mean","cvar95","cvar99",
                    "train_frozen_exceedance_threshold","exceedance_rate",
                    "mean_excess_given_exceedance","completion"]
    add("All TEST metrics finite",
        all(finite_series(T[c]) for c in numeric_cols),
        "checked " + ",".join(numeric_cols))
    add("All TEST n >= 20000", bool((pd.to_numeric(T["n"]) >= 20000).all()),
        sorted(pd.to_numeric(T["n"]).unique().tolist()))
    add("Completion in [0,1]",
        bool(((T["completion"] >= -TOL) & (T["completion"] <= 1+TOL)).all()),
        f"min={T['completion'].min():.12g}, max={T['completion'].max():.12g}")
    add("Exceedance rate in [0,1]",
        bool(((T["exceedance_rate"] >= -TOL) & (T["exceedance_rate"] <= 1+TOL)).all()),
        f"min={T['exceedance_rate'].min():.12g}, max={T['exceedance_rate'].max():.12g}")

    # Threshold must be asset-specific and frozen across every seed/objective.
    nthr = T.groupby("asset")["train_frozen_exceedance_threshold"].nunique()
    add("One frozen exceedance threshold per asset", bool((nthr == 1).all()), nthr.to_dict())

    # TEST gamma must equal TRAIN selection.
    sel = pd.DataFrame(train_lock.get("selection", []))
    gamma_ok = False
    gamma_detail = "missing selection"
    if {"asset","objective","gamma"}.issubset(sel.columns):
        z = T[["asset","objective","gamma"]].drop_duplicates()
        m = z.merge(sel[["asset","objective","gamma"]], on=["asset","objective"],
                    suffixes=("_test","_train"), how="outer", indicator=True)
        gamma_ok = bool((m["_merge"]=="both").all() and
                        np.allclose(m["gamma_test"],m["gamma_train"],rtol=0,atol=1e-12))
        gamma_detail = m.to_dict("records")
    add("TEST policies equal TRAIN-locked gamma", gamma_ok, gamma_detail)

    # Existing bootstrap file provenance checks.
    b_required = {"asset","objective","metric","estimate","ci025","ci975","n_seeds","bootstrap_reps"}
    add("TEST_BOOTSTRAP_CI required columns", b_required.issubset(B.columns),
        sorted(b_required - set(B.columns)) if not b_required.issubset(B.columns) else "all present")
    if b_required.issubset(B.columns):
        add("Existing bootstrap uses 15 seeds", bool((B["n_seeds"] == 15).all()),
            sorted(B["n_seeds"].unique().tolist()))
        add("Existing bootstrap reps >= 5000", bool((B["bootstrap_reps"] >= 5000).all()),
            sorted(B["bootstrap_reps"].unique().tolist()))

    passed = all(x["pass"] for x in checks)
    return {"version": VERSION, "overall_pass": passed, "checks": checks}


def build_summary(T, B):
    # Use direct seed means; carry the original I.6 bootstrap CI when available.
    rows = []
    for (asset,obj), g in T.groupby(["asset","objective"], sort=True):
        gamma = float(g["gamma"].iloc[0])
        for metric in METRICS:
            v = g.sort_values("seed")[metric].to_numpy(float)
            old = B[(B.asset==asset)&(B.objective==obj)&(B.metric==metric)]
            rows.append({
                "asset": asset, "objective": obj, "gamma": gamma, "metric": metric,
                "estimate": float(v.mean()),
                "sd_across_seeds": float(v.std(ddof=1)),
                "ci025_i6": float(old.iloc[0].ci025) if len(old)==1 else np.nan,
                "ci975_i6": float(old.iloc[0].ci975) if len(old)==1 else np.nan,
                "n_seeds": len(v),
            })
    return pd.DataFrame(rows)


def paired_analysis(T):
    rows_seed = []
    rows_agg = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scipy_ok = False
    try:
        from scipy.stats import wilcoxon
        scipy_ok = True
    except Exception:
        wilcoxon = None

    for asset in EXPECTED_ASSETS:
        A = T[T.asset==asset].copy()
        for comp in COMPARATORS:
            for metric in METRICS:
                p = A[A.objective==PRIMARY].set_index("seed")[metric]
                c = A[A.objective==comp].set_index("seed")[metric]
                common = sorted(set(p.index).intersection(c.index))
                pv = p.loc[common].to_numpy(float)
                cv = c.loc[common].to_numpy(float)
                d = pv - cv  # Localized - comparator. Negative is better for cost/risk metrics.
                if not LOWER_IS_BETTER[metric]:
                    improvement = d
                else:
                    improvement = -d

                for seed,pp,cc,dd,ii in zip(common,pv,cv,d,improvement):
                    rows_seed.append({
                        "asset":asset, "comparator":comp, "metric":metric, "seed":int(seed),
                        "localized":float(pp), "comparator_value":float(cc),
                        "delta_localized_minus_comparator":float(dd),
                        "improvement_signed_positive_is_better":float(ii),
                    })

                lo,hi = paired_bootstrap(d, BOOTSTRAP_REPS, rng)
                nneg=int(np.sum(d < -TOL)); npos=int(np.sum(d > TOL)); ntie=len(d)-nneg-npos
                wp=np.nan
                if scipy_ok and np.any(np.abs(d)>TOL):
                    try:
                        wp=float(wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                                          method="auto").pvalue)
                    except Exception:
                        wp=np.nan

                dm=float(d.mean())
                comp_mean=float(cv.mean())
                loc_mean=float(pv.mean())
                better = (d < -TOL) if LOWER_IS_BETTER[metric] else (d > TOL)
                rows_agg.append({
                    "asset":asset, "comparator":comp, "metric":metric,
                    "localized_estimate":loc_mean, "comparator_estimate":comp_mean,
                    "mean_delta_localized_minus_comparator":dm,
                    "relative_delta_pct":safe_pct(dm,comp_mean),
                    "paired_ci025":lo, "paired_ci975":hi,
                    "share_seeds_localized_better":float(np.mean(better)),
                    "n_better":int(np.sum(better)), "n_ties":ntie, "n_seeds":len(d),
                    "sign_test_p_two_sided":exact_sign_p(nneg,npos),
                    "wilcoxon_p_two_sided":wp,
                    "bootstrap_reps":BOOTSTRAP_REPS,
                })
    return pd.DataFrame(rows_seed), pd.DataFrame(rows_agg)


def reliability_table(T, paired):
    metrics = ["exceedance_rate","mean_excess_given_exceedance"]
    base = T.groupby(["asset","objective"],as_index=False).agg(
        gamma=("gamma","first"),
        exceedance_rate=("exceedance_rate","mean"),
        mean_excess_given_exceedance=("mean_excess_given_exceedance","mean"),
        completion=("completion","mean"),
        frozen_threshold=("train_frozen_exceedance_threshold","first"),
    )
    return base


def heterogeneity_table(paired):
    x = paired[paired.metric.isin(["mean","cvar95","cvar99","exceedance_rate",
                                    "mean_excess_given_exceedance"])].copy()
    # Descriptive cross-asset contrast, never pooled into a single efficacy claim.
    wide = x.pivot_table(index=["comparator","metric"], columns="asset",
                         values=["mean_delta_localized_minus_comparator",
                                 "relative_delta_pct","share_seeds_localized_better"])
    wide.columns = ["__".join(map(str,c)) for c in wide.columns]
    return wide.reset_index()


def make_figures(summary, paired, reliability, outdir):
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)

    labels = {
        "expected_cost":"Expected cost",
        "nominal_cvar":"Nominal CVaR",
        "global_kl":"Global-KL",
        "localized_kl":"Localized-KL",
    }

    for metric, fname, ylabel in [
        ("mean","fig_test_mean_cost.pdf","Mean implementation shortfall (bps)"),
        ("cvar95","fig_test_cvar95.pdf","CVaR95 implementation shortfall (bps)"),
    ]:
        z=summary[summary.metric==metric]
        assets=list(EXPECTED_ASSETS)
        objs=list(EXPECTED_OBJECTIVES)
        x=np.arange(len(assets),dtype=float)
        width=0.18
        fig,ax=plt.subplots(figsize=(8.2,4.8))
        for j,o in enumerate(objs):
            q=z[z.objective==o].set_index("asset").reindex(assets)
            vals=q["estimate"].to_numpy(float)
            lo=q["ci025_i6"].to_numpy(float); hi=q["ci975_i6"].to_numpy(float)
            err=np.vstack([vals-lo,hi-vals])
            ax.bar(x+(j-1.5)*width,vals,width,label=labels[o],
                   yerr=err,capsize=3)
        ax.set_xticks(x,["BTC","ETH"])
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False,ncol=2)
        ax.grid(axis="y",alpha=.2)
        fig.tight_layout()
        fig.savefig(outdir/fname,bbox_inches="tight")
        plt.close(fig)

    z=paired[paired.metric=="cvar95"].copy()
    comps=list(COMPARATORS)
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    ypos=[]; labs=[]; k=0
    for asset in EXPECTED_ASSETS:
        for comp in comps:
            r=z[(z.asset==asset)&(z.comparator==comp)].iloc[0]
            ax.errorbar(r["mean_delta_localized_minus_comparator"],k,
                        xerr=np.array([[r["mean_delta_localized_minus_comparator"]-r["paired_ci025"]],
                                       [r["paired_ci975"]-r["mean_delta_localized_minus_comparator"]]]),
                        fmt="o",capsize=3)
            ypos.append(k); labs.append(f"{asset.replace('USDT','')} vs {labels[comp]}")
            k+=1
    ax.axvline(0,linewidth=1)
    ax.set_yticks(ypos,labs)
    ax.set_xlabel("Paired seed-level difference in CVaR95: Localized-KL − comparator (bps)")
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout()
    fig.savefig(outdir/"fig_paired_policy_effects.pdf",bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.2,4.8))
    for asset in EXPECTED_ASSETS:
        z=reliability[reliability.asset==asset]
        ax.scatter(z["exceedance_rate"],z["mean_excess_given_exceedance"],
                   label=asset.replace("USDT",""))
        for _,r in z.iterrows():
            ax.annotate(labels[r.objective],(r.exceedance_rate,r.mean_excess_given_exceedance),
                        xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set_xlabel("Exceedance probability relative to frozen TRAIN threshold")
    ax.set_ylabel("Mean excess conditional on exceedance (bps)")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(outdir/"fig_reliability_decomposition.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    global BOOTSTRAP_REPS
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", default=r"C:\Users\fredy\Downloads\Results_RESS_V1I6_TURBO",
                    help="V1-I.6-TURBO results directory")
    ap.add_argument("--output", default=None,
                    help="Output directory (default: <input>/Results_RESS_V1I7)")
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args=ap.parse_args()

    BOOTSTRAP_REPS=int(args.bootstrap_reps)
    if BOOTSTRAP_REPS < 5000:
        raise SystemExit("--bootstrap-reps must be >= 5000")

    root=Path(args.input)
    out=Path(args.output) if args.output else root/"Results_RESS_V1I7"
    audit_dir=out/"audit"; table_dir=out/"tables"; fig_dir=out/"figures"; latex_dir=out/"latex"
    for d in (audit_dir,table_dir,fig_dir,latex_dir): d.mkdir(parents=True,exist_ok=True)

    required_files=[
        "TEST_SEED_METRICS.csv","TEST_BOOTSTRAP_CI.csv","TEST_COMPLETE.json",
        "TRAIN_LOCK.json","VALIDATION_LOCK.json","I6_MANIFEST.json"
    ]
    missing=[f for f in required_files if not (root/f).exists()]
    if missing:
        raise SystemExit("Missing required I.6 files: "+", ".join(missing))

    T=pd.read_csv(root/"TEST_SEED_METRICS.csv")
    B=pd.read_csv(root/"TEST_BOOTSTRAP_CI.csv")
    complete=load_json(root/"TEST_COMPLETE.json")
    train_lock=load_json(root/"TRAIN_LOCK.json")
    val_lock=load_json(root/"VALIDATION_LOCK.json")
    manifest=load_json(root/"I6_MANIFEST.json")

    audit=audit_inputs(root,T,B,complete,train_lock,val_lock,manifest)
    dump_json(audit_dir/"I7_INTEGRITY_AUDIT.json",audit)
    pd.DataFrame(audit["checks"]).to_csv(audit_dir/"I7_INTEGRITY_AUDIT.csv",index=False)
    if not audit["overall_pass"]:
        failed=[x["check"] for x in audit["checks"] if not x["pass"]]
        raise SystemExit("I.7 fail-closed integrity audit FAILED: "+"; ".join(failed))

    summary=build_summary(T,B)
    seed_diff, paired=paired_analysis(T)
    reliability=reliability_table(T,paired)
    heterogeneity=heterogeneity_table(paired)

    summary.to_csv(table_dir/"I7_TEST_SUMMARY.csv",index=False)
    seed_diff.to_csv(table_dir/"I7_PAIRED_DIFFERENCES.csv",index=False)
    paired.to_csv(table_dir/"I7_PAIRED_BOOTSTRAP_CI.csv",index=False)
    reliability.to_csv(table_dir/"I7_RELIABILITY_DECOMPOSITION.csv",index=False)
    heterogeneity.to_csv(table_dir/"I7_ASSET_HETEROGENEITY.csv",index=False)

    # Compact publication tables.
    pub = summary[summary.metric.isin(["mean","cvar95","cvar99","exceedance_rate"])].copy()
    pub["estimate_ci"] = pub.apply(
        lambda r: f"{r.estimate:.6g} [{r.ci025_i6:.6g}, {r.ci975_i6:.6g}]", axis=1)
    pub = pub[["asset","objective","gamma","metric","estimate_ci"]]
    write_latex_table(pub, latex_dir/"tab_i7_test_summary.tex",
                      "Out-of-sample TEST performance across the 15 predeclared seeds. "
                      "Intervals are the original V1-I.6 seed-level bootstrap 95\\% confidence intervals.",
                      "tab:i7_test_summary")

    pp=paired[paired.metric.isin(["mean","cvar95","cvar99","exceedance_rate"])][
        ["asset","comparator","metric","mean_delta_localized_minus_comparator",
         "relative_delta_pct","paired_ci025","paired_ci975",
         "share_seeds_localized_better","sign_test_p_two_sided","wilcoxon_p_two_sided"]
    ]
    write_latex_table(pp, latex_dir/"tab_i7_paired_inference.tex",
                      "Paired seed-level comparison of Localized-KL against the locked comparator policies. "
                      "Differences are Localized-KL minus comparator; negative differences favor Localized-KL "
                      "for cost and risk metrics.",
                      "tab:i7_paired_inference")

    write_latex_table(reliability, latex_dir/"tab_i7_reliability.tex",
                      "Reliability decomposition using the asset-specific Expected-Cost VaR95 threshold "
                      "frozen on TRAIN before TEST access.",
                      "tab:i7_reliability")

    make_figures(summary,paired,reliability,fig_dir)

    # Machine-readable headline results, without imposing a cross-asset pooled conclusion.
    headline=[]
    for asset in EXPECTED_ASSETS:
        for metric in ["mean","cvar95","cvar99","exceedance_rate","mean_excess_given_exceedance"]:
            r=paired[(paired.asset==asset)&(paired.comparator=="expected_cost")&
                     (paired.metric==metric)].iloc[0]
            headline.append({
                "asset":asset,"metric":metric,
                "comparison":"localized_kl_minus_expected_cost",
                "delta":float(r.mean_delta_localized_minus_comparator),
                "relative_delta_pct":None if pd.isna(r.relative_delta_pct) else float(r.relative_delta_pct),
                "paired_ci95":[float(r.paired_ci025),float(r.paired_ci975)],
                "share_seeds_localized_better":float(r.share_seeds_localized_better),
            })

    final={
        "version":VERSION,
        "status":"COMPLETE",
        "simulation_performed":False,
        "policy_retuning_performed":False,
        "input_directory":str(root),
        "integrity_audit_pass":True,
        "inference_unit":"seed",
        "paired_design":"same predeclared seeds / CRN-compatible seed-level comparison",
        "paired_bootstrap_reps":BOOTSTRAP_REPS,
        "primary_policy":PRIMARY,
        "comparators":list(COMPARATORS),
        "headline_vs_expected_cost":headline,
        "notes":[
            "Assets are analyzed separately; no pooled BTC/ETH efficacy claim is constructed.",
            "Paired bootstrap confidence intervals are primary for policy differences.",
            "Sign and Wilcoxon tests are secondary diagnostics and are not used for p-value-based model selection.",
            "Reliability exceedance thresholds remain frozen from TRAIN and are never re-estimated on TEST."
        ]
    }
    dump_json(out/"I7_FINAL_SUMMARY.json",final)

    print("="*72)
    print("RESS V1-I.7 COMPLETE — NO NEW SIMULATION / NO RETUNING")
    print("Input :",root)
    print("Output:",out)
    print("Integrity audit: PASS")
    print(f"Paired bootstrap: {BOOTSTRAP_REPS:,} resamples over 15 seeds")
    print("="*72)
    # Useful console headline: localized vs expected-cost.
    show=paired[(paired.comparator=="expected_cost") &
                (paired.metric.isin(["mean","cvar95","cvar99","exceedance_rate"]))]
    for _,r in show.iterrows():
        print(f"{r.asset:7s} {r.metric:16s} delta={r.mean_delta_localized_minus_comparator:+.8g} "
              f"({r.relative_delta_pct:+.3f}%) "
              f"CI95=[{r.paired_ci025:+.8g},{r.paired_ci975:+.8g}] "
              f"better={100*r.share_seeds_localized_better:.1f}% seeds")


if __name__=="__main__":
    main()
