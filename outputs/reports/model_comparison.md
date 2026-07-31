# Model Comparison Report

**Best Model:** Random Forest

## Selection Criteria

The best model is selected using a composite score that weights:
- RMSE (40%)
- MAE (35%)
- R² (15%)
- Training time (10%)

## Leaderboard

| Rank | Model | RMSE | MAE | R² | Train Time (s) | Predict Time (s) |
|------|-------|------|-----|----|----------------|------------------|
| 1 | Random Forest ⭐ | $49,417 | $32,203 | 0.8136 | 7.598 | 0.0713 |
| 2 | Gradient Boosting | $56,467 | $38,765 | 0.7567 | 2.386 | 0.0089 |
| 3 | Decision Tree | $67,437 | $42,981 | 0.6530 | 0.205 | 0.0030 |
| 4 | Linear Regression | $69,944 | $50,498 | 0.6267 | 0.022 | 0.0031 |
| 5 | Lasso Regression | $69,944 | $50,498 | 0.6267 | 0.310 | 0.0023 |
| 6 | Ridge Regression | $69,951 | $50,504 | 0.6266 | 0.043 | 0.0024 |
| 7 | SVR | $116,845 | $86,969 | -0.0419 | 5.602 | 2.1665 |

## Summary

The **Random Forest** model achieved the best balance of accuracy (RMSE=$49,417, MAE=$32,203, R²=0.8136) and training efficiency among all evaluated algorithms.
