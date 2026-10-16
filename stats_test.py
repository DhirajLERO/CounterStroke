import numpy as np
import pandas as pd
from scipy.stats import rankdata, shapiro, ttest_rel, wilcoxon

BASE = "Cifar10/outputs/generation/per_sample_metrics.csv"
CS = "Cifar10/graco/results/unstretched_per_sample_metrics.csv"

base = pd.read_csv(BASE)
cs = pd.read_csv(CS)
cs = cs[cs["method"].isin(["graco_base_o2", "graco_base_o3", "graco50"])].copy()

def class_flip(df):
    pred = pd.to_numeric(df["pred_class_cf"], errors="coerce")
    ok = (df["has_valid_cf"].fillna(0).astype(int) == 1) & pred.notna()
    return ok & (pred.astype("Int64") != df["y_true"].astype("Int64"))

base["flip"] = class_flip(base)
grad = base[base["method"] == "counterfactual_pt"].set_index("sample_idx")
grad_ok = grad.index[grad["flip"]]

def rank_biserial(diff):
    nonzero = diff[np.asarray(diff) != 0]
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))

def paired(winner, other, metric):
    common = winner.index.intersection(other.index)
    a = winner.loc[common, metric].to_numpy(float)
    b = other.loc[common, metric].to_numpy(float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    diff = a - b
    spread = float(np.std(diff, ddof=1))
    shapiro_p = float(shapiro(diff).pvalue)
    if shapiro_p >= 0.05:
        p = float(ttest_rel(a, b).pvalue)
        effect = float(diff.mean() / spread)
        kind, test = "d_z", "paired_t"
    else:
        p = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        effect = rank_biserial(diff)
        kind, test = "r", "wilcoxon"
    print(
        f"n={len(a)}  {test}  p={p:.3e}  {kind}={effect:+.2f}  "
        f"means {a.mean():.4f} vs {b.mean():.4f}"
    )

sparse = cs[cs["method"] == "graco_base_o2"].set_index("sample_idx")
prox = cs[cs["method"] == "graco_base_o3"].set_index("sample_idx")

print("sparsity: CS_sparse vs GradCF")
paired(sparse.loc[sparse.index.intersection(grad_ok)], grad.loc[grad_ok], "sparsity_frac")
print("L2: GradCF vs CS_prox")
paired(grad.loc[grad_ok], prox.loc[prox.index.intersection(grad_ok)], "l2_distance")
