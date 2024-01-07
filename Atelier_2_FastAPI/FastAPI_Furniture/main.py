import joblib  # Use joblib for loading models
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Define a Pydantic model for input data
class FurnitureInput(BaseModel):
    category: str
    sellable_online: bool
    other_colors: str
    depth: float
    height: float
    width: float


app = FastAPI()

# Label encodings
label_encodings = {
    "category": {
        "Bar furniture": 0,
        "Beds": 1,
        "Bookcases & shelving units": 2,
        "Cabinets & cupboards": 3,
        "Café furniture": 4,
        "Chairs": 5,
        "Chests of drawers & drawer units": 6,
        "Children's furniture": 7,
        "Nursery furniture": 8,
        "Outdoor furniture": 9,
        "Room dividers": 10,
        "Sideboards, buffets & console tables": 11,
        "Sofas & armchairs": 12,
        "TV & media furniture": 13,
        "Tables & desks": 14,
        "Trolleys": 15,
        "Wardrobes": 16,
    },
    "sellable_online": {False: 0, True: 1},
    "other_colors": {"No": 0, "Yes": 1},
}


# Function to preprocess input data
def preprocess_data(input_data: FurnitureInput):
    input_data_dict = dict(input_data)
    input_data_dict["category"] = label_encodings["category"][input_data_dict["category"]]
    input_data_dict["sellable_online"] = label_encodings["sellable_online"][
        input_data_dict["sellable_online"]
    ]
    input_data_dict["other_colors"] = label_encodings["other_colors"][
        input_data_dict["other_colors"]
    ]

    df = pd.DataFrame([input_data_dict])
    return df


# Try loading the model
try:
    model = joblib.load("../models/DTmodel.pkl")  # Update the path if necessary

except Exception as e:
    print(f"Error loading the model: {e}")
    model = None


@app.post("/predict")
def predict(input_data: FurnitureInput):
    print(input_data)
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")


    data = preprocess_data(input_data)
    prediction = model.predict(data)
    return {"prediction": prediction[0]}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Furniture Prediction API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)