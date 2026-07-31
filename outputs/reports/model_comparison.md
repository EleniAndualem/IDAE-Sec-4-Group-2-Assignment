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
| 1 | Random Forest ⭐ | $50,476 | $32,960 | 0.8056 | 1.533 | 0.0720 |
| 2 | Gradient Boosting | $54,380 | $37,041 | 0.7743 | 6.741 | 0.0368 |
| 3 | Decision Tree | $67,437 | $42,981 | 0.6530 | 0.552 | 0.0317 |
| 4 | Linear Regression | $69,944 | $50,498 | 0.6267 | 0.218 | 0.0341 |
| 5 | Lasso Regression | $69,944 | $50,498 | 0.6267 | 10.838 | 0.0280 |
| 6 | Ridge Regression | $69,951 | $50,504 | 0.6266 | 0.099 | 0.0382 |
| 7 | SVR | $116,845 | $86,969 | -0.0419 | 30.367 | 16.7617 |

## Summary

The **Random Forest** model achieved the best balance of accuracy (RMSE=$50,476, MAE=$32,960, R²=0.8056) and training efficiency among all evaluated algorithms.
