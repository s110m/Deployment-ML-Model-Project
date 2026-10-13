"""
Train the churn prediction model and save it to disk.

This is the "notebook turned into a script" version of section 2 of
`model_deployment.ipynb`:

    1. load and clean the Telco churn dataset
    2. validate the model with K-fold cross-validation
    3. train the final model on the full training set and check it on the test set
    4. save (DictVectorizer, LogisticRegression) to `model_C=1.0.bin` with pickle

Usage:
    pipenv run python train.py
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DATA_PATH = 'data/churn_telco_data.csv'
C = 1.0                 # inverse regularisation strength of LogisticRegression
N_SPLITS = 5            # number of folds for cross-validation
RANDOM_STATE = 1
OUTPUT_FILE = f'model_C={C}.bin'

NUMERICAL = ['tenure', 'monthlycharges', 'totalcharges']
CATEGORICAL = [
    'gender', 'seniorcitizen', 'partner', 'dependents',
    'phoneservice', 'multiplelines', 'internetservice',
    'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
    'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
    'paymentmethod',
]
FEATURES = CATEGORICAL + NUMERICAL


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_data(path):
    """Read the CSV and normalise column names and string values."""
    df = pd.read_csv(path)

    # 'Monthly Charges' -> 'monthly_charges'
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # 'Fiber optic' -> 'fiber_optic' in every text column
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    # totalcharges has blanks for brand-new customers -> treat them as 0
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)

    # target: 1 = the customer left, 0 = stayed
    df['churn'] = (df['churn'] == 'yes').astype(int)
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train(df_train, y_train, C=1.0):
    """One-hot encode the features with DictVectorizer and fit a logistic regression."""
    dicts = df_train[FEATURES].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)
    return dv, model


def predict(df, dv, model):
    """Return the churn probability for every row of `df`."""
    dicts = df[FEATURES].to_dict(orient='records')
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]


def cross_validate(df_full_train, C, n_splits):
    """AUC on each fold of a K-fold split."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        dv, model = train(df_train, df_train['churn'].values, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(df_val['churn'].values, y_pred)
        scores.append(auc)
        print(f'  fold {fold}: AUC = {auc:.3f}')

    return scores


def main():
    df = load_data(DATA_PATH)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    print(f'Cross-validating with C={C}, {N_SPLITS} folds')
    scores = cross_validate(df_full_train, C, N_SPLITS)
    print(f'C={C}  AUC = {np.mean(scores):.3f} +- {np.std(scores):.3f}')

    print('Training the final model on the full training set')
    dv, model = train(df_full_train, df_full_train['churn'].values, C=C)
    y_pred = predict(df_test, dv, model)
    print(f'Test AUC = {roc_auc_score(df_test["churn"].values, y_pred):.3f}')

    # save both the vectorizer and the model: we need the same encoding at prediction time
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f'Model saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
