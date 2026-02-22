# Telecom Customer Churn and Retention

## Section 1 - Project Summary
Built a customer churn prediction model for a telecommunications dataset and created a retention simulation to maximize profit. The final model achieved a PR-AUC of 0.673 (26.5% churn rate) and generated an estimated $28.7k in expected retention profit from 506 targeted customers.

## Section 2 - Data and Problem Statement
Given a dataset representing customer subscription and billing data from a telecom provider, predict customer churn. Then, identify high-risk customers and deploy a retention strategy to optimize retention spending and future customer value.

## Section 3 - Modeling Approach
Used a preprocessing pipeline with one-hot encoding for categorical variables and standard scaling for numerical variables. Four models were trained and evaluated: logistic regression, random forest, XGBoost, and gradient boosting (sklearn GBM). The models were tuned to optimize PR-AUC because of the class imbalance (26.5% churn rate), as PR-AUC better reflects performance on the minority churn class than accuracy. Hyperparameters were carefully adjusted to reduce overfitting.

To reduce overfitting:
- Tree-based models had limited depth, larger minimum samples per leaf, subsampling, and feature sampling.
- Boosting models used lower learning rates and early stopping to improve generalization.
- Logistic regression was regularized (L2) to provide a stable, interpretable baseline.

## Section 4 - Modeling Results and Evaluation
The models produced these results:
| Model                | Train PR-AUC | Test PR-AUC | Gap   | F1    |
|----------------------|-------------|-------------|-------|-------|
| GBM (sklearn)        | 0.720       | 0.674       | 0.047 | 0.644 |
| XGBoost              | 0.740       | 0.672       | 0.068 | 0.639 |
| Random Forest        | 0.693       | 0.659       | 0.034 | 0.625 |
| Logistic Regression  | 0.686       | 0.646       | 0.041 | 0.628 |

Key Insights:
- Across the four models, GBM (sklearn) and XGBoost had the highest PR-AUC. Of the two, GBM demonstrated a smaller generalization gap while maintaining comparable PR-AUC, making it the more stable final choice.
- Logistic regression has slightly less predictive power than the other models but it provides a stable and interpretable baseline.
- Random Forest had the lowest train-test gap but underperformed in PR-AUC and F1.

For these reasons, I decided to use GBM (sklearn) as my final model.

## Section 5 - Retention and Business Impact
After selecting the GBM model, I began the retention simulation with the following assumptions:
- Retention outreach has a 20% success rate.
- Retention cost is $50 per targeted customer.
- Successfully retained customers will stay with the company for 12 months with the same monthly charges they previously had.
The expected profit formula was:
Expected Profit = P(churn) × P(save) × Customer Value − Retention Cost
where customer value = monthly charges x 12.

The retention simulation targeted 502 high-risk customers and generated an estimated $28,574.21 in expected profit. This project demonstrates the transition from predictive modeling to profit-driven decision optimization.