import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_test(y, treatment, confidence=0.95):
    """Difference in two independent proportions.

    Uses the pooled standard error for the equality-of-proportions z test and
    the unpooled standard error for the Wald confidence interval.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    if y.ndim != 1 or t.ndim != 1 or len(y) != len(t):
        raise ValueError("y and treatment must be one-dimensional arrays of equal length")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Outcome must be binary (0/1)")
    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("Treatment must be binary (0/1)")
    a, b = y[t == 1], y[t == 0]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both treatment and control groups must be non-empty")
    p1, p0 = float(a.mean()), float(b.mean())
    diff = p1 - p0
    pooled = float((a.sum() + b.sum()) / (len(a) + len(b)))
    se_test = np.sqrt(pooled * (1-pooled) * (1/len(a) + 1/len(b)))
    z = diff / se_test if se_test > 0 else 0.0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    se_ci = np.sqrt(p1*(1-p1)/len(a) + p0*(1-p0)/len(b))
    alpha = 1-confidence
    zcrit = float(stats.norm.ppf(1-alpha/2))
    ci = (diff-zcrit*se_ci, diff+zcrit*se_ci)
    return {
        "treated_rate": p1, "control_rate": p0, "ate": diff,
        "standard_error": float(se_ci), "test_standard_error": float(se_test),
        "z_statistic": float(z), "p_value": pvalue,
        "ci_lower": float(ci[0]), "ci_upper": float(ci[1]),
        "n_treated": int(len(a)), "n_control": int(len(b)),
    }


def mean_difference(y, treatment, confidence=0.95):
    """Welch two-sample difference in means with Satterthwaite CI."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    if y.ndim != 1 or t.ndim != 1 or len(y) != len(t):
        raise ValueError("y and treatment must be one-dimensional arrays of equal length")
    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("Treatment must be binary (0/1)")
    a, b = y[t == 1], y[t == 0]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group must contain at least two observations")
    diff = float(a.mean() - b.mean())
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    term_a, term_b = var_a/len(a), var_b/len(b)
    se = float(np.sqrt(term_a + term_b))
    df_num = (term_a + term_b)**2
    df_den = (term_a**2/(len(a)-1)) + (term_b**2/(len(b)-1))
    dof = df_num/df_den if df_den else len(a)+len(b)-2
    alpha = 1-confidence
    q = float(stats.t.ppf(1-alpha/2, dof))
    pvalue = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return {
        "treated_mean": float(a.mean()), "control_mean": float(b.mean()),
        "ate": diff, "standard_error": se, "degrees_of_freedom": float(dof),
        "p_value": pvalue, "ci_lower": diff-q*se, "ci_upper": diff+q*se,
        "n_treated": int(len(a)), "n_control": int(len(b)),
    }


def standardized_mean_difference(x1, x0):
    x1, x0 = np.asarray(x1, dtype=float), np.asarray(x0, dtype=float)
    pooled = np.sqrt((np.var(x1, ddof=1)+np.var(x0, ddof=1))/2)
    return float((np.mean(x1)-np.mean(x0))/pooled) if pooled else 0.0


def randomized_assignment_diagnostics(df, treatment_col="treatment", features=None):
    """Summarize baseline balance for a randomized treatment assignment.

    For numeric covariates this reports standardized mean differences (SMD).
    For categorical covariates, one-hot level indicators are assessed. These are
    diagnostics of the observed sample, not a test that randomization was valid.
    """
    if treatment_col not in df:
        raise ValueError(f"Missing treatment column: {treatment_col}")
    if features is None:
        features = [c for c in df.columns if c != treatment_col]
    treatment = np.asarray(df[treatment_col], dtype=int)
    if treatment.ndim != 1 or not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("Treatment column must be binary (0/1)")
    if len(np.unique(treatment)) != 2:
        raise ValueError("Both treatment and control observations are required")

    treated = df.loc[treatment == 1]
    control = df.loc[treatment == 0]
    rows = []
    for feature in features:
        if feature not in df:
            raise ValueError(f"Missing covariate: {feature}")
        series = df[feature]
        if series.isna().any():
            raise ValueError(f"Covariate {feature} contains missing values")
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 2:
            x1 = treated[feature].astype(float).to_numpy()
            x0 = control[feature].astype(float).to_numpy()
            pooled_sd = np.sqrt((np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2)
            smd = float((np.mean(x1) - np.mean(x0)) / pooled_sd) if pooled_sd else 0.0
            rows.append({"feature": feature, "level": None, "smd": smd,
                         "abs_smd": abs(smd), "kind": "numeric"})
        else:
            levels = sorted(series.astype(str).unique().tolist())
            for level in levels:
                x1 = (treated[feature].astype(str) == level).astype(float).to_numpy()
                x0 = (control[feature].astype(str) == level).astype(float).to_numpy()
                pooled_sd = np.sqrt((np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2)
                smd = float((x1.mean() - x0.mean()) / pooled_sd) if pooled_sd else 0.0
                rows.append({"feature": feature, "level": level, "smd": smd,
                             "abs_smd": abs(smd), "kind": "indicator"})
    return {
        "n_treated": int((treatment == 1).sum()),
        "n_control": int((treatment == 0).sum()),
        "rows": rows,
        "max_abs_smd": max((r["abs_smd"] for r in rows), default=0.0),
        "share_abs_smd_below_0_1": float(np.mean([r["abs_smd"] < 0.1 for r in rows])) if rows else 1.0,
    }


def naive_observed_difference(y, treatment, confidence=0.95):
    """Unadjusted observed difference in means, returned with a clear label.

    In a randomized experiment this is also the design-based difference-in-means
    estimator (subject to the experiment's assignment and sampling assumptions).
    In observational data it is only an associational contrast, not a causal
    estimator without exchangeability/overlap and a defensible adjustment design.
    """
    result = mean_difference(y, treatment, confidence=confidence)
    result["interpretation"] = "unadjusted observed difference; causal interpretation depends on design assumptions"
    return result


def placebo_permutation_test(y, treatment, n_permutations=2000, seed=42):
    """Permutation diagnostic under the sharp null of no treatment effect.

    This is a randomization-style check, not proof of no confounding and not a
    substitute for documenting the actual assignment mechanism.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    if y.ndim != 1 or t.ndim != 1 or len(y) != len(t) or len(y) < 2:
        raise ValueError("y and treatment must be equal-length 1D arrays")
    if not np.isfinite(y).all() or not set(np.unique(t)).issubset({0, 1}) or len(np.unique(t)) != 2:
        raise ValueError("Finite outcomes and both binary treatment arms are required")
    if n_permutations < 100:
        raise ValueError("Use at least 100 permutations")
    observed = float(y[t == 1].mean() - y[t == 0].mean())
    rng = np.random.default_rng(seed)
    permuted = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_t = rng.permutation(t)
        permuted[i] = y[permuted_t == 1].mean() - y[permuted_t == 0].mean()
    pvalue = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (n_permutations + 1))
    return {"observed_difference": observed, "p_value_two_sided": pvalue,
            "n_permutations": int(n_permutations), "seed": int(seed),
            "null_quantiles": np.quantile(permuted, [0.025, 0.5, 0.975]).tolist()}
