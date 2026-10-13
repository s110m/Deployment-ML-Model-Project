"""
The smallest possible web service: GET /ping -> "pong".

Run:   python ping.py
Test:  open http://localhost:9696/ping in a browser
       or  curl http://localhost:9696/ping
"""

from flask import Flask

app = Flask('ping')


# The decorator registers `ping` as the handler for the URL path /ping.
# The path can be anything ('/ping', '/hello', ...); `methods` lists the HTTP
# methods it accepts - GET is what a browser sends when you open a URL.
@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'


if __name__ == '__main__':
    # host='0.0.0.0' -> listen on all network interfaces, not only localhost
    app.run(debug=True, host='0.0.0.0', port=9696)
