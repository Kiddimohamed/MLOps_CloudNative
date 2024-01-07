import joblib  # Use joblib for loading models
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

# Define a Pydantic model for input data
class MPIInput(BaseModel):
    MeshSize: float
    Compiler: float
    NumberofVariables: float
    VariableType: float
    NumberofNodes: float
    NumberofCores: float
    Dim: float



app = FastAPI()


# Try loading the model



@app.post("/predict")
def predict(input_data: MPIInput):
    model = joblib.load("../models/knn_over.pkl")  # Update the path if necessary

    input_data_dict = dict(input_data)
    print(type(input_data_dict))
    data = pd.DataFrame([input_data_dict])

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    prediction = model.predict(data)
    print(prediction[0])
    return {"prediction": str(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Furniture Prediction API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)