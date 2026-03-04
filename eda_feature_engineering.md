# Supplementary Document: EDA and Feature Engineering

This document provides detailed descriptions of the feature selection techniques, encoding functions, and UID 
construction used in the project. All methods in this section are adapted from the competition winners
[Chris Deotte and Konstantin Yakovlev](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2).

## V-Feature Reduction

Since the number of V features was large, they were reduced by grouping columns with similar NaN structure, then 
applying PCA to each group. A maximum sized subset of uncorrelated columns was selected from each group, and the 
average of the entire group was also used as a feature.

## Feature Selection

The following feature selection techniques were applied:

### Forward Feature Selection
Features were incrementally added (single or grouped), keeping only those that improved model performance. This 
helped identify the minimal set of features that maximized predictive power, which was especially valuable with 400+ 
mostly anonymized features where intuition could not guide selection.

### Recursive Feature Elimination
Starting with all features, the least useful ones were iteratively removed (single or grouped). This complemented 
forward selection by approaching from the opposite direction, catching features that seemed individually weak but 
contributed through interactions.

### Permutation Importance
Each feature was shuffled and the resulting drop in model performance was measured. This provided a model-agnostic 
importance ranking that accounted for how the model actually used each feature, rather than relying on proxy metrics 
like information gain.

### Adversarial Validation
A classifier was trained to distinguish train from test rows; features that made this easy indicated distribution 
drift. Since the competition data was split chronologically, some features shifted between train and test periods, and 
including them would cause the model to overfit to patterns that no longer held.

### Correlation Analysis
Redundant features that were highly correlated with each other were identified and removed. With many anonymized 
V-columns, redundancy was significant, and removing it reduced noise and improved model stability.

### Time Consistency
For each feature (or small group of features), a single model was trained on the first month of the training data and 
used to predict fraud on the last month. This evaluated whether each feature's predictive signal was stable over time. 
Roughly 95% of features were consistent, but 5% showed train AUC around 0.60 and validation AUC around 0.40, meaning 
they captured patterns that actively reversed in the future.

### Client Consistency
Each feature's values were checked for stability across transactions from the same client or card. Features that 
fluctuated randomly for the same client were likely noise rather than meaningful signal, and removing them reduced 
overfitting.

### Train/Test Distribution Analysis
Feature distributions were directly compared between train and test to flag shifts. This served as a complementary 
check alongside adversarial validation, providing a more visual and interpretable view of distribution drift across the 
time-based split.

## Encoding Functions

Five encoding functions were used to transform raw features into model-ready representations.

### Frequency Encoding (encode_FE)
Each categorical value was replaced with its frequency of occurrence, calculated across the combined train and test 
sets. Encoding on the combined data ensured consistent frequency estimates and prevented information leakage from 
train/test distribution differences.

### Label Encoding (encode_LE)
Categorical features were converted to integer labels using factorization across the combined train and test sets. 
Memory-efficient dtypes (int16 or int32) were chosen automatically based on the number of unique values.

### Aggregation Encoding (encode_AG)
Numeric features were aggregated (mean, std) within groups defined by identifier columns such as card or address. This 
captured how a transaction compared to the typical behavior of its associated group, for example, whether a transaction 
amount was unusual for a given card.

### Feature Combination (encode_CB)
Two columns were concatenated into a single string feature and then label encoded. This created interaction features 
that represented specific combinations, such as a particular card used at a particular address.

### Nunique Aggregation (encode_AG2)
For each group, the number of unique values of a given feature was counted. This captured diversity within a group, for 
example, how many distinct email domains were associated with a single card.

## UID Construction

Since the dataset was anonymized with no explicit client identifier, UIDs were constructed by combining `card1_addr1` 
(a previously created interaction feature of card number and address) with a derived reference date 
(`floor(day - D1)`). The intuition is that `card1_addr1` narrows down a card at a specific address, while `day - D1` 
produces a value that remains constant across transactions for the same client, acting as a proxy for an account start 
date. Together these create a fingerprint that approximates a client identity. The UID is imperfect, as some UID values 
contain multiple clients, but the tree-based models can detect this and further split these groups using additional 
features.

Once constructed, the UID enabled a large set of group aggregation features that captured client-level behavior. These 
included the mean and standard deviation of numeric features like `TransactionAmt` and D-columns per UID, the mean of 
C-columns and M-columns per UID, and nunique counts measuring how many distinct values of features like 
`P_emaildomain`, `C13`, and various V-columns appeared per UID. These aggregations give the model context for whether a 
transaction looks normal relative to the client's history, for example, whether a transaction amount is unusually high 
for that client, or whether an unfamiliar email domain is being used.