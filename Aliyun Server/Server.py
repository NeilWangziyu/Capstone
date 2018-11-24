from flask import Flask, request, send_from_directory
from flask import request
from flask_cors import *
import json

jsndir = '/root/result/'
fname = 'result.json'
app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    print('hello!')
    return send_from_directory(jsndir, fname, as_attachment=False)


# @app.route('/', methods=['GET', 'POST'])
# def sendjson():
#   print('Connected!Congradulation!')
#  return send_from_directory(jsndir, fname, as_attachment=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)