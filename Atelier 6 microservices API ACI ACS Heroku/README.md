# Weather API Project

## Overview
This project is a Flask-based API that provides weather information based on user input. Users can interact with the API through a Flask form, and the project is containerized using Docker for easy deployment and testing. Additionally, the Docker image is uploaded to Azure Container Registry for seamless integration with cloud services.

## Features
* **Flask API**: Built using Flask, the API retrieves weather information based on user input.
* **Flask Form**: Users can input relevant data through a user-friendly form provided by Flask.
* **Dockerized**: The project is containerized using Docker, ensuring consistent deployment across environments.
* **Azure Integration**: The Docker image is uploaded to Azure Container Registry for cloud-based deployment.
## How to Use
### Prerequisites
```
* Docker installed on your machine.
* Azure Container Registry account for uploading the Docker image.
```
### Building the Project


1. Once our application hosted on heroku, we will be ready to consume it using the following [link](https://ewinsou-ml-api-dep-0c58759ae3ed.herokuapp.com)


2. install the project. 
```
cd app_consuming
pip install requirements.txt
python weather_app.py
```
Now connect to [127.0.0.1:8000](127.0.0.1:8000) and your application is ready for testing.

![image](https://media.discordapp.net/attachments/1184809383683706960/1184813119726436443/image.png?ex=658d5634&is=657ae134&hm=3696c62263368496a953af0513cd28e3584f723736f67024cd63b8e5a0129f13&=&format=webp&quality=lossless&width=935&height=499)

3. Build the Docker image:
we can easly build the docker image using the following code
```
docker build -t weather-api:latest .
```

Now let's open our docker application, we will have our docker image is built.
![image](https://media.discordapp.net/attachments/1184809383683706960/1184813541681811516/image.png?ex=658d5699&is=657ae199&hm=01408a354a0736be297c81ddcebec8c0a9d8afacdb9912a786fa00f9ef7ad8ce&=&format=webp&quality=lossless&width=742&height=499)

click on run and we will have our application is working on the [127.0.01](127.0.0.1)

![image](https://media.discordapp.net/attachments/1184809383683706960/1184813120359759992/image.png?ex=658d5634&is=657ae134&hm=a1c777da9dfa96430ed72f5f0156f429f84d28d6fd4a0627a0137d987fe0ae30&=&format=webp&quality=lossless&width=935&height=499)

![image](https://media.discordapp.net/attachments/1184809383683706960/1184813273263120445/image.png?ex=658d5659&is=657ae159&hm=3269305f81403a46472286c5d9c64ca8ba167aa5e66b29f6a71bd2f749c77186&=&format=webp&quality=lossless&width=742&height=499)

## Uploading to Azure Container Registry


1. Install and Log in to Azure Container Registry:
Fist let's install azure CLI
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```
you may more instructions about the installation by using the following [link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt)

Now let login into our account. 'kiddioubrahim' is container registry and it must be created before running this command.
```
az acr login --use-device-code
az acr login -n kiddioubrahim

```
![image](https://media.discordapp.net/attachments/1184809383683706960/1184883015256645782/image.png?ex=658d974c&is=657b224c&hm=e3f5b2d8327561c8324107fc2f36442e893308da03b83a5c7c6875c424d97403&=&format=webp&quality=lossless&width=1152&height=306)

we can see that we are logged successfuly   
2. Tag the Docker image and push the image:
```
docker tag fastapi:latest kiddioubrahim.azurecr.io/fastapi:latest
docker push kiddioubrahim.azurecr.io/fastapi

```

![image](https://media.discordapp.net/attachments/1184809383683706960/1184883016267464714/Screenshot_from_2023-12-14_15-41-00.png?ex=658d974d&is=657b224d&hm=7bd4798aa9b059474781b5d09d36047c3f47b578b5f5771e5d42d1c485843a8d&=&format=webp&quality=lossless&width=1152&height=450)

&#x26A0; **authentification problem**: it seems that we are facing problem with authentification since we're using uca mail for this project.

## Azure MLflow Deploy

Now let's upload the 'Azure Mlflow Deploy' Folder to azure ml, and run it.

![image](https://media.discordapp.net/attachments/1184809383683706960/1186064846895186001/image.png?ex=6591e3f7&is=657f6ef7&hm=16a42a140ebc6ada701a6fa6c58c94d6531448820a117ae498885d4779341369&=&format=webp&quality=lossless&width=887&height=499)

One the notebook is done. we can navigate to jobs and ovserve the our models summary.

![iamge](https://media.discordapp.net/attachments/1184809383683706960/1185682676355702814/Screenshot_from_2023-12-16_20-30-04.png?ex=6590800a&is=657e0b0a&hm=f8849bf88ea7fb2738ba4b3da195c1f0991fe72e57d909dbdaba76bbd1afb944&=&format=webp&quality=lossless&width=887&height=499)

We can also observe our model parametres summary
![image](https://media.discordapp.net/attachments/1184809383683706960/1185682675953041588/Screenshot_from_2023-12-16_20-30-26.png?ex=6590800a&is=657e0b0a&hm=64c62a1086c11b414f02db640f8c411273d8a9fa8b715ecce2b57c375fb1e340&=&format=webp&quality=lossless&width=887&height=499)

![image](https://media.discordapp.net/attachments/1184809383683706960/1185682677366534154/Screenshot_from_2023-12-16_20-29-37.png?ex=6590800b&is=657e0b0b&hm=af605c35961a6556db8f0062092f4836fa2d66e4b9b6938af551cbcb85e5f788&=&format=webp&quality=lossless&width=887&height=499)

* to deploy our we can add an endpoint and deploy it.

![image](https://media.discordapp.net/attachments/1184809383683706960/1186062339586408499/image.png?ex=6591e1a1&is=657f6ca1&hm=5a892440c608b4cfdb96aa9e8f697738ee125006df43548a919bd8c61ef9d60d&=&format=webp&quality=lossless&width=887&height=499)

as you can see, we are running out of ressource. that's why can't deploy our model.
