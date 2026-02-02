import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def cohend(sample_group1, sample_group2):
  """
  Compute Cohen's d for the standardized mean difference between two independent groups.

  Cohen's d is defined as:
    d = (mean(sample_group1) - mean(sample_group2)) / s_pooled
  where s_pooled is the pooled standard deviation of the two groups.

  Parameters
  ----------
  sample_group1 : array-like
    Sample values for group 1 (e.g., exam scores for males). Must be numeric.
  sample_group2 : array-like
    Sample values for group 2 (e.g., exam scores for females). Must be numeric.

  Returns
  -------
  float
    Cohen's d effect size. Positive values indicate group 1 has a higher mean than group 2.

  Notes
  -----
  - Assumes the two samples are independent.
  - Uses the unbiased sample variance (ddof=1) for each group and a pooled standard deviation.
  - If either group has fewer than 2 observations, or if the pooled standard deviation is 0,
    the result may be undefined (division by zero).

  Examples
  --------
  >>> male_scores = np.array([80, 75, 90])
  >>> female_scores = np.array([70, 72, 68])
  >>> cohend(male_scores, female_scores)
  2.236...
  """

  # Calculate the size of the samples
  n1, n2 = len(sample_group1), len(sample_group2)

  # Calculate the variance of the samples
  s1, s2 = np.var(sample_group1, ddof=1), np.var(sample_group2, ddof=1)

  # Calculate the pooled standard deviation
  s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

  # Calculate the means of the samples
  u1, u2 = np.mean(sample_group1), np.mean(sample_group2)

  # Calculate the effecti size
  return (u1 - u2) / s_pooled

def anova_eta_squared(df: pd.DataFrame, outcome_col: str, group_col: str):
  """
  Run a one-way ANOVA (via OLS) for a continuous outcome across a categorical group,
  and compute eta-squared (η²) as an effect size.

  This function fits the model:
      outcome_col ~ C(group_col)
  using Ordinary Least Squares (OLS), then produces a Type-II ANOVA table and η².

  Eta-squared is defined as:
      η² = SS_between / SS_total
  where SS_between is the sum of squares attributable to the grouping factor and
  SS_total = SS_between + SS_residual.

  Parameters
  ----------
  df : pandas.DataFrame
    Input dataframe containing the outcome and group columns.
  outcome_col : str
    Name of the numeric outcome column (e.g., "exam_score").
  group_col : str
    Name of the categorical grouping column (e.g., "gender", "course").

  Returns
  -------
  anova_table : pandas.DataFrame
    Statsmodels ANOVA table (Type-II) with sum of squares, degrees of freedom,
    F-statistic, and p-value for the grouping factor.
  eta_sq : float
    Eta-squared (η²), the proportion of total variance in the outcome explained
    by the grouping factor.

  Notes
  -----
  - Rows with missing values in `outcome_col` or `group_col` are dropped.
  - This uses classic one-way ANOVA assumptions (independent observations,
    approximately normal residuals, and equal variances across groups).
    With very large samples, the p-value can be extremely small even when η² is tiny.
  - If you suspect unequal variances and/or highly imbalanced group sizes,
    consider Welch’s ANOVA for the hypothesis test, and report an appropriate effect size.

  Examples
  --------
  >>> anova_tbl, eta2 = anova_eta_squared(df, "exam_score", "gender")
  >>> print(anova_tbl)
  >>> print(eta2)
  """
  df = df.dropna(subset=[outcome_col, group_col]).copy()

  # 1) One-way ANOVA via Ordinary Least Squares (OLS)
  model = smf.ols(f"{outcome_col} ~ C({group_col})", data=df).fit()
  anova_table = sm.stats.anova_lm(model, typ=2)  # typ=2 is standard for one factor

  # 2) Eta-squared (η²) = SS_between / SS_total
  factor_row = f"C({group_col})"
  ss_between = anova_table.loc[factor_row, "sum_sq"]
  ss_resid = anova_table.loc["Residual", "sum_sq"]
  ss_total = ss_between + ss_resid
  eta_sq = ss_between / ss_total

  return anova_table, eta_sq
