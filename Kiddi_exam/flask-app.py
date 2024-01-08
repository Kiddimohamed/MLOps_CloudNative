from flask import Flask, render_template, request
import joblib


app = Flask(__name__)

from pydantic import BaseModel

class ValidateData(BaseModel):
    review:str



@app.route('/', methods=['Get', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    data = request.form
    d = ValidateData(**data)
    filename = 'Models/LR_model.pkl'
    loaded_model = joblib.load(open(filename, 'rb'))
    # transofrm
    vec_file = 'Models/fitted_vectorizer.pkl'
    vectorizer = joblib.load(open(vec_file, 'rb'))
    model_input = vectorizer.transform([data['review']])
    res = loaded_model.predict(model_input)[0]
    return render_template('index.html', sentiment=res, review=data['review'])



app.run(debug=True)