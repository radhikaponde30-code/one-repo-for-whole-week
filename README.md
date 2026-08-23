# Statistical Analysis and Hypothesis Testing — Car Features & MSRP

Week 3 internship task: statistical hypothesis testing in Python on a cleaned
car-listings dataset (11,199 rows, from Week 1 data cleaning).

## Business Questions Tested

| # | Question | Test | Result |
|---|----------|------|--------|
| H1 | Does transmission type affect price? | Welch's t-test + Mann-Whitney U | Rejected H0 (p < 0.0001); auto ≈ $11,900 higher than manual |
| H2 | Does vehicle size affect price? | One-way ANOVA + Tukey HSD | Rejected H0 (p < 0.0001); all size pairs differ |
| H3 | Are size and transmission related? | Chi-square test of independence | Rejected H0 (p < 0.0001); Cramer's V = 0.32 |
| H4 | Does horsepower predict price? | Simple linear regression (OLS) | Rejected H0 (p < 0.0001); R² = 0.44 |

## Files

- `analysis.py` — full analysis script (data prep, all 4 hypothesis tests, visualizations)
- `cars_clean.csv` — cleaned dataset used for this analysis (see Week 1 report for cleaning methodology)
- `results.json` — key statistics from each test (means, p-values, effect sizes, confidence intervals)
- `viz1_transmission_boxplot.png` — MSRP by transmission type
- `viz2_qqplot.png` — normality check (Q-Q plots, raw vs. log-transformed MSRP)
- `viz3_anova_violin.png` — MSRP distribution by vehicle size
- `viz4_chisquare_stacked.png` — transmission type share by vehicle size
- `viz5_regression_fit.png` — horsepower vs. price regression fit

## How to Run

```bash
pip install pandas numpy scipy seaborn matplotlib
python analysis.py
```

## Full Write-Up

See the accompanying Word report (`Week3_Statistical_Analysis_Report.docx`) for
full methodology, assumption checks, and discussion of results.
