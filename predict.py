"""
Churn prediction web service (Flask).

Endpoints
    GET  /          health check -> {"status": "ok"}  (used by Elastic Beanstalk)
    POST /predict   body: one customer as JSON -> {"churn": bool, "churn_probability": float}

Run it
    development (any OS):  pipenv run python predict.py
    production (Linux):    gunicorn --bind=0.0.0.0:9696 predict:app
    production (Windows):  waitress-serve --listen=0.0.0.0:9696 predict:app
"""

import pickle
from pathlib import Path

from flask import Flask, jsonify, request

MODEL_FILE = Path(__file__).parent / 'model_C=1.0.bin'
CHURN_THRESHOLD = 0.5   # probability above which we treat the customer as "will churn"

# Load the model once, when the service starts - not on every request.
with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn-service')


def predict_single(customer):
    """Churn probability for one customer given as a dict of features."""
    X = dv.transform([customer])
    return model.predict_proba(X)[0, 1]


@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json(silent=True)
    if not isinstance(customer, dict):
        return jsonify({'error': 'send one customer as a JSON object'}), 400

    churn_probability = predict_single(customer)

    result = {
        # numpy types (np.float64, np.bool_) are converted to plain Python types
        # so they can be serialised to JSON
        'churn_probability': float(churn_probability),
        'churn': bool(churn_probability >= CHURN_THRESHOLD),
    }
    return jsonify(result)


if __name__ == '__main__':
    # Flask's built-in server: fine for development, not for production
    app.run(debug=True, host='0.0.0.0', port=9696)
