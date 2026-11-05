import numpy as np
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
