# Adult Dataset: Ingestion, Validation, Preprocessing, and Model Evaluation

## 1. Data Ingestion

The Adult dataset was loaded using the `fetch_openml()` function without modifying its original contents. No cleaning or transformation operations were performed during ingestion, ensuring that the raw data remained unchanged for subsequent processing.

The dataset consists of **48,842 rows and 15 columns**. These records were used as the starting point for the validation, preprocessing, and machine learning stages.

## 2. Data Validation

Schema validation was performed to verify the dataset's structure, data types, numerical ranges, missing values, and duplicate records.

### 2.1 Data Type Checks

The following numerical columns were checked against their expected data types:

- age
- fnlwgt
- education-num
- capital-gain
- capital-loss
- 
hours-per-week

All six columns matched the expected data types, and no type violations were detected.

### 2.2 Range Checks

Range validation was performed on four numerical features to identify values outside the predefined limits.

| Feature |         Expected range      | Violations |
| age                  17–90                  0 
| hours-per-week        1–99                  0 
| capital-gain        0–99,999                0 
| capital-loss        0–4,356                 0 

The results confirmed that all four features satisfied their specified range constraints. No out-of-range values were identified.

### 2.3 Missing Value Checks

The dataset was examined for missing values represented by `?`. No such values were detected, and the validation process reported zero missing-value violations under this check.

Missing-value imputation was nevertheless included in the preprocessing pipeline to provide a consistent approach to handling missing data.

### 2.4 Duplicate Checks

Duplicate detection identified **52 duplicate rows** in the original dataset. These records were retained during validation to preserve the raw data. Their removal was deferred to the cleaning stage.

## 3. Data Cleaning and Transformation

The dataset was divided into training and testing subsets using a **70/30 stratified split**. Stratification was used to preserve the class distribution as closely as possible across the two subsets.

Imputation was performed using statistics calculated exclusively from the training data. This prevented information from the test set from influencing the preprocessing parameters.

The following strategies were applied:

- **Numerical features:** Missing values were replaced with the training-set median
- **Categorical features:** Missing values were replaced with the most frequent category in the training set

This approach reduced the risk of data leakage and maintained a clear separation between model training and evaluation.

The pipeline also included outlier handling and duplicate removal as part of the cleaning process.

## 4. Feature Engineering

Two additional features were created to investigate whether combining existing variables could improve Logistic Regression performance.

### 4.1 Net Capital

The `capital_net` feature represents the difference between capital gains and capital losses.

`capital_net = capital-gain - capital-loss`

This feature combines the two capital-related variables into a single measure of net capital gain or loss.

### 4.2 Hours Worked Relative to Age

The `hours_per_age` feature was calculated as follows:

`hours_per_age = hours-per-week / (age - 16)`

This feature expresses weekly working hours relative to age beyond 16 years.

To evaluate the contribution of these features, a Logistic Regression model was trained and assessed both before and after feature engineering. The same evaluation metrics were used to compare the two configurations.

## 5. Model Evaluation

Model performance was evaluated using accuracy and F1 score. The baseline model used the original features, while the second model included the two engineered features.

| Metric | Baseline model | Model with engineered features |
|---|---:|---:|
| Accuracy | 0.8456 | 0.8452 |
| F1 score | 0.6339 | 0.6330 |

### 5.1 Baseline Results

The Logistic Regression baseline achieved an accuracy of **0.8456** and an F1 score of **0.6339**. These values served as the reference for evaluating the effect of feature engineering.

### 5.2 Results After Feature Engineering

After adding `capital_net` and `hours_per_age`, the model achieved an accuracy of **0.8452** and an F1 score of **0.6330**.

Both metrics decreased slightly:

- Accuracy decreased by 0.0004
- F1 score decreased by 0.0009

### 5.3 Discussion

The results indicate that the engineered features did not improve the model's predictive performance in this experiment. The original feature set achieved marginally better results on both evaluation metrics.

One possible explanation is that the original variables already captured much of the information represented by the derived features. As a result, the additional features may not have contributed sufficient new information to improve classification.

However, the small differences should not be interpreted as proof that feature engineering is ineffective in general. Further experiments with alternative feature representations and classification algorithms could provide additional insight.

Based on the observed results, the baseline model performed slightly better than the model incorporating the engineered features.

## 6. Label Quality Assessment

A duplicate-conflict audit was conducted to identify records that had identical feature values but different target labels. The dataset was grouped by feature values, and the corresponding class labels were examined for inconsistencies.

The audit identified **five duplicate-conflict groups**. These groups suggest potential label inconsistencies, as identical input features were associated with different class labels.

Such conflicts can make classification more challenging because the model may encounter identical feature patterns associated with different outcomes. The findings therefore highlight possible data quality issues that may require further investigation.

### 6.1 Random Sample Inspection

A random sample of 50 records was selected and saved to:

`results/sample_50_rows.csv`

The sampled records were reviewed against the Adult dataset documentation to assess whether any impossible values were present. No impossible values were identified in the inspected sample.

Although this review provided an additional data quality check, the results apply only to the sampled records and do not establish that the entire dataset is free from inconsistencies.

## 7. Reproducibility

The complete data processing and evaluation pipeline can be executed using a single command:

```bash
python lab04/src/pipeline.py
```

The pipeline automates the following operations:

1. Data ingestion
2. Schema and data validation
3. Data cleaning
4. Missing-value imputation
5. Outlier handling
6. Duplicate removal
7. Feature engineering
8. Logistic Regression evaluation
9. Label quality auditing
10. SHA-256 hashing

Automating these stages improves consistency and simplifies the execution of the experiment. SHA-256 hashing provides a mechanism for verifying file integrity by comparing hash values to detect changes in file contents.

For reproducible results, the dataset version, random seed, preprocessing configuration, and model evaluation settings should also be documented and kept consistent between runs.

## 8. Conclusion

The Adult dataset was processed through a structured pipeline covering data ingestion, validation, cleaning, feature engineering, model evaluation, and data quality auditing.

The validation stage confirmed that the selected numerical features met their expected data types and predefined range constraints. No values represented by `?` were detected, while 52 duplicate rows were identified for subsequent handling during cleaning.

The Logistic Regression baseline achieved an accuracy of 0.8456 and an F1 score of 0.6339. Adding the two engineered features resulted in slightly lower scores of 0.8452 and 0.6330, indicating that these transformations did not improve performance in the current experiment.

The label quality audit identified five groups with conflicting class labels among records with identical feature values, highlighting potential inconsistencies in the dataset. Additionally, the inspection of a random sample of 50 records revealed no impossible values.

Finally, the automated pipeline and SHA-256 hashing support consistent execution and data integrity verification. Overall, the experiment demonstrates the importance of systematic validation, leakage-aware preprocessing, empirical feature evaluation, and data quality assessment when developing machine learning models.