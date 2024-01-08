# Kiddi Test
## Overview
This project involves creating, training, and deploying machine learning models using MLflow, FastAPI, Flask, Docker, and Azure services. The process includes data preprocessing, model training, tracking model performance, saving the best model in ONNX format, creating REST APIs, packaging models and applications as Docker containers, and deploying them using various platforms like Heroku, Azure Container Instances, and Azure ML SDK.

Steps:
1. Preprocessing
Begin by preprocessing your dataset to prepare it for model training.
2. Model Training
Train five machine learning models using the preprocessed data.
3. Tracking Model Performance
Use MLflow to track models' performance, versions, and parameters during training.
4. Saving Best Model
Save the best performing model in ONNX format.
Serialize the preprocessing transformations using the transformers API and save it in pickle format.
5. FastAPI Implementation
Utilize FastAPI to create a REST API for your model.
6. Dockerization
Package your model and FastAPI application as a Docker container.
7. Postman Consumption
Use Postman to consume and test the created APIs.
8. Flask Application
Develop a dedicated Flask application to consume your created API.
9. Dockerize Flask Application
Package the Flask application as a Docker container.
Bonus:
1. Heroku Deployment
Deploy your API using Heroku.
2. Azure Container Instances Deployment
Deploy your containerized API using Azure Container Instance.
3. Azure ML SDK and Mlflow for Model Deployment
Deploy your model as a service using Azure ML SDK and Mlflow.
