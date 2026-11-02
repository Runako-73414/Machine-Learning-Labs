import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import hashlib

adult = fetch_openml(
name="adult",
version=2,
as_frame=True
)
df = adult.frame
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())

schema = {
    "age": "numeric",
    "workclass": "categorical",
    "fnlwgt": "numeric",
    "education": "categorical",
    "education-num": "numeric",
    "marital-status": "categorical",
    "occupation": "categorical",
    "relationship": "categorical",
    "race": "categorical",
    "sex": "categorical",
    "capital-gain": "numeric",
    "capital-loss": "numeric",
    "hours-per-week": "numeric",
    "native-country": "categorical",
    "class": "categorical"
}

print("\n=== TYPE CHECKS ===")

for col, expected in schema.items():

    if expected == "numeric":
        print(col, "->", df[col].dtype)

print("\n=== RANGE CHECKS ===")

print(
    "Age violations:",
    ((df["age"] < 17) | (df["age"] > 90)).sum()
)

print(
    "Hours violations:",
    ((df["hours-per-week"] < 1) |
     (df["hours-per-week"] > 99)).sum()
)

print(
    "Capital gain violations:",
    ((df["capital-gain"] < 0) |
     (df["capital-gain"] > 99999)).sum()
)

print(
    "Capital loss violations:",
    ((df["capital-loss"] < 0) |
     (df["capital-loss"] > 4356)).sum()
)

print("\n=== MISSING VALUES ===")
for col in df.columns:
    missing = (df[col] == "?").sum()
    print(col, ":", missing)

print("\n=== DUPLICATES ===")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates) 

report = f"""
VALIDATION REPORT

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Age violations: {((df["age"] < 17) | (df["age"] > 90)).sum()}

Hours violations:
{((df["hours-per-week"] < 1) | (df["hours-per-week"] > 99)).sum()}

Capital gain violations:
{((df["capital-gain"] < 0) | (df["capital-gain"] > 99999)).sum()}

Capital loss violations:
{((df["capital-loss"] < 0) | (df["capital-loss"] > 4356)).sum()}

Duplicate rows:
{df.duplicated().sum()}
"""

with open(
    "lab04/results/validation_report.txt",
    "w"
) as f:

    f.write(report)

print("\nValidation report saved.")

X = df.drop("class", axis=1)

y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

numeric_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

numeric_imputer = SimpleImputer(
    strategy="median"
)

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

numeric_imputer.fit(
    X_train[numeric_features]
)

categorical_imputer.fit(
    X_train[categorical_features]
)

X_train[numeric_features] = numeric_imputer.transform(
    X_train[numeric_features]
)

X_test[numeric_features] = numeric_imputer.transform(
    X_test[numeric_features]
)

X_train[categorical_features] = categorical_imputer.transform(
    X_train[categorical_features]
)

X_test[categorical_features] = categorical_imputer.transform(
    X_test[categorical_features]
)

print("\nImputation completed.")

for col in numeric_features:

    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)

    X_train[col] = X_train[col].clip(lower, upper)

    X_test[col] = X_test[col].clip(lower, upper)

print("Outlier clipping completed.")

before_rows = len(X_train)

train_combined = X_train.copy()

train_combined["class"] = y_train.values

train_combined = train_combined.drop_duplicates()

after_rows = len(train_combined)

print(
    "Train duplicates removed:",
    before_rows - after_rows
)

X_train = train_combined.drop(
    "class",
    axis=1
)

y_train = train_combined["class"]

preprocessor = {
    "numeric_imputer": numeric_imputer,
    "categorical_imputer": categorical_imputer
}

joblib.dump(
    preprocessor,
    "lab04/results/preprocessor.joblib"
)

print("Preprocessor saved.")

baseline_train = X_train.copy()
baseline_test = X_test.copy()

baseline_train = pd.get_dummies(
    baseline_train
)

baseline_test = pd.get_dummies(
    baseline_test
)

baseline_train, baseline_test = baseline_train.align(
    baseline_test,
    join="left",
    axis=1,
    fill_value=0
)

baseline_model = LogisticRegression(
max_iter=1000
)
baseline_model.fit(
baseline_train,
y_train
)
baseline_pred = baseline_model.predict(
baseline_test
)
baseline_acc = accuracy_score(
y_test,
baseline_pred
)
baseline_f1 = f1_score(
y_test,
baseline_pred,
pos_label=">50K"
)
print("\nBASELINE RESULTS")
print("Accuracy:", baseline_acc)
print("F1:", baseline_f1)

engineered_train = X_train.copy()
engineered_test = X_test.copy()

engineered_train["capital_net"] = (
    engineered_train["capital-gain"]
    - engineered_train["capital-loss"]
)

engineered_test["capital_net"] = (
    engineered_test["capital-gain"]
    - engineered_test["capital-loss"]
)

engineered_train["hours_per_age"] = (
    engineered_train["hours-per-week"]
    / (engineered_train["age"] - 16)
)

engineered_test["hours_per_age"] = (
    engineered_test["hours-per-week"]
    / (engineered_test["age"] - 16)
)

engineered_train = pd.get_dummies(
    engineered_train
)

engineered_test = pd.get_dummies(
    engineered_test
)

engineered_train, engineered_test = engineered_train.align(
    engineered_test,
    join="left",
    axis=1,
    fill_value=0
)

feature_model = LogisticRegression(
max_iter=1000
)
feature_model.fit(
engineered_train,
y_train
)
feature_pred = feature_model.predict(
engineered_test
)
feature_acc = accuracy_score(
y_test,
feature_pred
)
feature_f1 = f1_score(
y_test,
feature_pred,
pos_label=">50K"
)
print("\nFEATURE ENGINEERING RESULTS")
print("Accuracy:", feature_acc)
print("F1:", feature_f1)

audit_df = df.copy()

feature_columns = list(
    audit_df.drop("class", axis=1).columns
)

conflict_groups = (
    audit_df.groupby(
        feature_columns,
        observed=True
    )["class"]
    .nunique()
)

conflict_count = (conflict_groups > 1).sum()

print(
    "\nDuplicate-conflict rows:",
    conflict_count
)

sample_50 = df.sample(
    n=50,
    random_state=42
)

sample_50.to_csv(
    "lab04/results/sample_50_rows.csv",
    index=False
)

print("Random 50-row sample saved.")

processed_data = pd.concat(
    [X_train, y_train],
    axis=1
)

csv_string = processed_data.to_csv(
    index=False
)

dataset_hash = hashlib.sha256(
    csv_string.encode()
).hexdigest()

print("SHA256:", dataset_hash)

with open(
    "lab04/results/dataset_hash.txt",
    "w"
) as f:

    f.write(dataset_hash)