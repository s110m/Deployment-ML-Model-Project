"""
Client for the churn prediction service: send one customer, act on the answer.

Usage
    python predict_client.py                                   # local Flask / waitress (port 9696)
    python predict_client.py --host localhost:8080             # Docker with -p 8080:9696
    python predict_client.py --host <env>.elasticbeanstalk.com # AWS Elastic Beanstalk
    python predict_client.py --customer my_customer.json       # a different customer
"""

import argparse
import json

import requests


def main():
    parser = argparse.ArgumentParser(description='Ask the churn service about one customer.')
    parser.add_argument('--host', default='localhost:9696',
                        help='host[:port] of the service (default: localhost:9696)')
    parser.add_argument('--customer', default='customer.json',
                        help='JSON file with the customer features (default: customer.json)')
    args = parser.parse_args()

    with open(args.customer, encoding='utf-8') as f_in:
        customer = json.load(f_in)

    url = f'http://{args.host}/predict'
    response = requests.post(url, json=customer, timeout=10)
    response.raise_for_status()
    result = response.json()

    print(f'POST {url}')
    print(f'Response: {result}')

    # the business decision that uses the model's answer
    if result['churn']:
        print(f"Churn probability {result['churn_probability']:.1%} -> send a discount offer")
    else:
        print(f"Churn probability {result['churn_probability']:.1%} -> no action needed")


if __name__ == '__main__':
    main()
