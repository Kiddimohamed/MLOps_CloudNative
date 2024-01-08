from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")


class ValidateData(BaseModel):
    review: str


@app.get('/Test')
def home(request: Request):
    data = None
    return "welcome!"


@app.post('/predict')
def predict(data: ValidateData):
    review = data.review
    filename = 'Models/LR_model.pkl'
    loaded_model = joblib.load(open(filename, 'rb'))

    # transform
    vectorizer = joblib.load(open('Models/fitted_vectorizer.pkl', 'rb'))
    try:
        model_input = vectorizer.transform([review])
        res = loaded_model.predict(model_input)[0]
    except Exception as e:
        print(e)
        res = 1  # Handle the exception appropriately
    return str(res)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
