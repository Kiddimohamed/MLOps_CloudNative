# This is a sample Python script.

import pandas as pd
from flask import Flask, request, render_template
from predict import pred
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

app = Flask(__name__)
Datadup = pd.read_excel("Data/res_new.xlsx")


@app.route('/')
def index():

    return render_template('index.html',data= None)


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == "POST":
        result = pred(Datadup ,request.form.to_dict())
        data = request.form.to_dict()

    else:
        result = None
        data = None
    return render_template('index.html', prediction=result, data=data)





if __name__ == '__main__':
    app.run(debug=True)
