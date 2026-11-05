import numpy as np
from scipy import stats

def two_proportion_test(y, treatment):
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    a, b = y[t == 1], y[t == 0]
    p1, p0 = float(a.mean()), float(b.mean())
    diff = p1 - p0
    se = np.sqrt(p1*(1-p1)/len(a) + p0*(1-p0)/len(b))
    z = diff / se if se > 0 else 0.0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    ci = (diff - 1.96*se, diff + 1.96*se)
    return {"treated_rate": p1, "control_rate": p0, "ate": diff,
            "standard_error": float(se), "z_statistic": float(z),
            "p_value": pvalue, "ci_lower": float(ci[0]), "ci_upper": float(ci[1]),
            "n_treated": int(len(a)), "n_control": int(len(b))}

def mean_difference(y, treatment, confidence=0.95):
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    a, b = y[t == 1], y[t == 0]
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)))
    df_num = (a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))**2
    df_den = ((a.var(ddof=1)/len(a))**2/(len(a)-1) +
              (b.var(ddof=1)/len(b))**2/(len(b)-1))
    dof = df_num/df_den if df_den else len(a)+len(b)-2
    q = stats.t.ppf(1-(1-confidence)/2, dof)
    return {"treated_mean": float(a.mean()), "control_mean": float(b.mean()),
            "ate": diff, "standard_error": se, "p_value": float(stats.ttest_ind(a,b,equal_var=False).pvalue),
            "ci_lower": diff-float(q*se), "ci_upper": diff+float(q*se),
            "n_treated": int(len(a)), "n_control": int(len(b))}

def standardized_mean_difference(x1, x0):
    x1, x0 = np.asarray(x1, dtype=float), np.asarray(x0, dtype=float)
    pooled = np.sqrt((np.var(x1, ddof=1)+np.var(x0, ddof=1))/2)
    return float((np.mean(x1)-np.mean(x0))/pooled) if pooled else 0.0
