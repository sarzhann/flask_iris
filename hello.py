from flask import Flask, request, jsonify, abort, redirect, url_for
app = Flask(__name__)

import numpy as np
import joblib

clf = joblib.load('knn.pkl')

@app.route('/')
def hello_world():
    print(3*9)
    return '<h1>Hello, my lovely Madina!</h1>'

@app.route('/user/<username>')
def show_user_profile(username):
    name = username.upper()
    return f'User name is {name}'

@app.route('/avg/<nums>')
def avg(nums):
    nums = list(map(float, nums.split(',')))
    avg = sum(nums)/len(nums)
    return f'average = {avg}'

@app.route('/iris/<values>')
def predict(values):

    values = np.array(list(map(float, values.split(',')))).reshape(1,-1)
    pred = clf.predict(values)[0]

    flowers = {0:'setosa',1:'versicolor',2:'virginica'}
    flower = flowers[pred]

    return f'<h1>{flower}</h1><img src = "/static/{flower}.jpg", alt="{flower}">'

@app.route('/badrequest400')
def bad_request():
    return abort(400)

@app.route('/iris_post', methods = ['POST'])
def add_message():

    try:
        content = request.get_json()

        values = np.array(list(map(float, content['flower'].split(',')))).reshape(1,-1)
        pred = clf.predict(values)[0]

        flowers = {0:'setosa',1:'versicolor',2:'virginica'}
        flower = {'class':str(pred), 'flower':flowers[pred]}

    except:
        return redirect(url_for('bad_request'))

    return jsonify(flower)