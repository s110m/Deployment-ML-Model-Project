"""
Predict churn for one customer *without* a web service.

This is the first step of deployment: the notebook code turned into a script
that loads the saved model and scores a customer. It only works on the
machine where it runs - section 5.3 turns it into a web service.

Usage:
    pipenv run python predict_offline.py
"""

import json
import pickle

MODEL_FILE = 'model_C=1.0.bin'
CHURN_THRESHOLD = 0.5

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict(customer):
    """Return (will_churn, churn_probability) for one customer dict."""
    X = dv.transform([customer])
    churn_probability = model.predict_proba(X)[0, 1]
    return bool(churn_probability >= CHURN_THRESHOLD), float(churn_probability)


if __name__ == '__main__':
    with open('customer.json', encoding='utf-8') as f_in:
        customer = json.load(f_in)

    will_churn, churn_probability = predict(customer)
    print(f'churn probability = {churn_probability:.3f}  ->  churn = {will_churn}')
