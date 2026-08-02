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
| 1 | Random Forest ⭐ | $50,476 | $32,960 | 0.8056 | 0.371 | 0.0162 |
| 2 | Gradient Boosting | $54,410 | $37,051 | 0.7741 | 1.983 | 0.0071 |
| 3 | Decision Tree | $67,437 | $42,981 | 0.6530 | 0.206 | 0.0032 |
| 4 | Linear Regression | $69,944 | $50,498 | 0.6267 | 0.021 | 0.0033 |
| 5 | Lasso Regression | $69,944 | $50,498 | 0.6267 | 0.209 | 0.0026 |
| 6 | Ridge Regression | $69,951 | $50,504 | 0.6266 | 0.019 | 0.0030 |
| 7 | SVR | $116,845 | $86,969 | -0.0419 | 5.558 | 2.1656 |

## Summary

The **Random Forest** model achieved the best balance of accuracy (RMSE=$50,476, MAE=$32,960, R²=0.8056) and training efficiency among all evaluated algorithms.
