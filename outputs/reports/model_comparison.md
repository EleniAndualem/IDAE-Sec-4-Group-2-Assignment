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
| 1 | Random Forest ⭐ | $50,476 | $32,960 | 0.8056 | 1.314 | 0.0624 |
| 2 | Gradient Boosting | $54,380 | $37,041 | 0.7743 | 6.209 | 0.0283 |
| 3 | Decision Tree | $67,437 | $42,981 | 0.6530 | 0.482 | 0.0626 |
| 4 | Linear Regression | $69,944 | $50,498 | 0.6267 | 0.265 | 0.0235 |
| 5 | Lasso Regression | $69,944 | $50,498 | 0.6267 | 9.146 | 0.0249 |
| 6 | Ridge Regression | $69,951 | $50,504 | 0.6266 | 0.118 | 0.0227 |
| 7 | SVR | $116,845 | $86,969 | -0.0419 | 27.448 | 13.9476 |

## Summary

The **Random Forest** model achieved the best balance of accuracy (RMSE=$50,476, MAE=$32,960, R²=0.8056) and training efficiency among all evaluated algorithms.
