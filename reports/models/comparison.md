# Model Comparison

## churn

| Model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| random_forest | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| xgboost | 0.9991 | 0.9986 | 1.0 | 0.9993 | 1.0 |

## occupancy

| Model | mae | rmse | mape |
|---|---|---|---|
| xgboost | 4.7684 | 7.7302 | 72.611 |
| prophet | 38.6283 | 40.5099 | 119.5186 |

## pricing

| Model | mae | rmse | mape |
|---|---|---|---|
| xgboost | 25.0586 | 32.5148 | 20.2389 |

## restaurant

| Model | mae | rmse | mape |
|---|---|---|---|
| breakfast | 112.7909 | 150.9059 | 61.4453 |
| lunch | 88.4663 | 116.4969 | 65.197 |
| dinner | 161.144 | 212.8634 | 60.8189 |

## staffing

| Model | mae | rmse | mape |
|---|---|---|---|
| regression | 0.7305 | 0.8623 | 6.5222 |
