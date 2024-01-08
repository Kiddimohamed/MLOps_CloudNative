# model Testing 
## Introduction 
### Performance Testing Lab
Welcome to the Performance Testing Lab! Here, we'll evaluate different deployment methods:

- FastAPI Deployment
- Docker Containerization
- TFX Model Serving

We'll start by serving a Question Answering model with FastAPI, containerize it using Docker for production efficiency, and explore TFX for optimized deep learning model serving. Load testing with Locust will help us analyze performance across these methods. Let's dive in!

## Serve an NLP model (transformer) using FastAPI
We'll deploy a Question Answering (QA) model via FastAPI using the "transformers" library, specifically utilizing DistillBERT, a lighter variant of BERT. Our focus is on a DistillBERT model pre-trained on the SQUAD dataset. In the "1. fastAPI_Transformer_model_serving" folder, access "main.py":

Create a Pydantic data model for inputs (Question-Answer).
Initialize FastAPI and load the pre-trained model via the "pipeline" module in the "transformers" library.
Set up an asynchronous function to establish an API endpoint.
Launch the API using uvicorn.
Execute the file with "python main.py" and explore the API at http://127.0.0.1:8000/docs to validate its functionality.
![](https://media.discordapp.net/attachments/1191490101247758479/1193691229997776927/Screenshot_from_2024-01-08_00-01-23.png?ex=65ada296&is=659b2d96&hm=a16d66da57dc90f8eb0fd34759e3bafa66c6fe2ffe18342e6563df527e49c667&=&format=webp&quality=lossless&width=720&height=644)

## Containerizing our API using Docker

For efficient production and simplified deployment, Docker is crucial. Isolating your service and application is vital, ensuring code execution consistency across any operating system. Access the "2. Dockerizing_API" folder for further details.
we need to create Docker file you our fastapi . 
![](https://media.discordapp.net/attachments/1191490101247758479/1193704843030822982/Screenshot_from_2024-01-08_00-56-30.png?ex=65adaf44&is=659b3a44&hm=11ee6b6bcc3a3d69dc5b51ababae33eb04ff55f40073c2fdfad3c32f46f73eb4&=&format=webp&quality=lossless&width=738&height=660)

and then we will build our container and run it before testing it with post man in the same way.
![](https://media.discordapp.net/attachments/1191490101247758479/1193707125998891179/Screenshot_from_2024-01-08_01-05-35.png?ex=65adb164&is=659b3c64&hm=11ccdf2fc5da62d45fa7d5613abd21e6e203ed96601df3cccd30f421f01c83d0&=&format=webp&quality=lossless)
![](https://media.discordapp.net/attachments/1191490101247758479/1193691229997776927/Screenshot_from_2024-01-08_00-01-23.png?ex=65ada296&is=659b2d96&hm=a16d66da57dc90f8eb0fd34759e3bafa66c6fe2ffe18342e6563df527e49c667&=&format=webp&quality=lossless&width=720&height=644)


## Serve a transformer type model using TFX
TFX streamlines deep learning model serving, offering speed and efficiency. However, a few crucial points need understanding before implementation. Models must be TensorFlow registered models for TFX utilization.
Open the “3. Faster_Transformer_model_serving_using_Tensorflow_Extended” folder and
run the “saved_model.ipynb” notebook to produce the saved model.

Firstly, we'll retrieve the Docker image for Tensorflow Extended
With the model prepared and accessible via TFX Docker, it can seamlessly integrate with another service. The necessity for an additional service arises from the specialized input format required by Transformer-based models, which necessitates text preprocessing before feeding it into the model.
To achieve this, create a fastAPI service that will invoke the API served by the TensorFlow serving container. Before initiating your service's code, ensure the Docker container is running with specified parameters for executing the BERT-based sentiment analysis model.

Please maintain this window open to sustain the service, as we'll be utilizing it with FastAPI. The overarching structure of the new API operates as follows:

Now that our TFX service utilizing Docker is primed for consumption via REST API on port 8501, we'll access it through FastAPI."


![](https://media.discordapp.net/attachments/1191490101247758479/1193727592541474836/b.png?ex=65adc474&is=659b4f74&hm=6bb9f4166808c1025d9fe02719e50ebd9b301bb894c6f7bd17fe2fa13e9a54b7&=&format=webp&quality=lossless)
## Loading Test using Locust

Numerous applications cater to load testing, offering valuable insights into service response time, latency, and failure rates. Among these tools, Locust stands out as one of the finest for this task. Our aim is to employ Locust to evaluate the load handling capabilities of three methods we've explored previously:

Utilizing fastAPI alone,
Employing Dockerized fastAPI, and
Serving the model via TFX using fastAPI.
Let's begin by installing Locus
To create a simulation, we utilize the HttpUser class, inheriting from it to define our User class. The @task decorator is crucial in delineating the specific task our user will execute. In this case, the 'predict' function represents the task, repeatedly generating a random string of 20 characters and forwarding it to your API.

To initiate the test, execute the command 'locust -f locust_file.py'. Access the testing interface at http://localhost:8089/. Upon reaching this link, you'll encounter the interface below.

Configure the simulation parameters as follows: set the total number of simulated users to 10, the spawn rate to 1, and designate the host as http://127.0.0.1:8000, where our service is hosted. Once these parameters are set, click 'Start swarming'.

The interface will update, signaling the commencement of testing. At any point, you can halt the test by clicking the 'Stop' button.
![](https://media.discordapp.net/attachments/1191490101247758479/1193732911137050634/Screenshot_from_2024-01-08_02-46-03.png?ex=65adc968&is=659b5468&hm=02754d032da091382904269468ca62c6f9ef24058ecf39fa52f80b47fd6201ea&=&format=webp&quality=lossless&width=880&height=660)
and then we will check the results :
![](https://media.discordapp.net/attachments/1191490101247758479/1193733299143725146/Screenshot_from_2024-01-08_02-49-31.png?ex=65adc9c5&is=659b54c5&hm=c0c6f9d18cddad4b65fb7056a006f33a2c0e0e219523a675ed408fe9d8695be7&=&format=webp&quality=lossless&width=671&height=660)


# Conclusion 

In conclusion, this project facilitates load testing for various API configurations using Locust. By leveraging the capabilities of Locust, users can simulate and assess the performance of different API setups, including those utilizing fastAPI, Dockerized environments, and TFX integration. The provided 'locust_file.py' defines the user behavior for these simulations, allowing for the generation of random data payloads to stress-test the endpoints.

By following the instructions outlined in the README, users can effortlessly initiate tests, configure parameters through the Locust web interface, and monitor service performance. This project aims to simplify the process of load testing while providing a versatile framework for evaluating API responsiveness and scalability