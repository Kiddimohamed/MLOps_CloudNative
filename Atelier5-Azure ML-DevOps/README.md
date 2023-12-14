
# Port of Turku Weather Forecasting with Azure DevOps, MLflow, and Azure Machine Learning SDK

This repository tackles a real-world challenge using Azure DevOps, MLflow, and the Azure Machine Learning SDK. We aim to forecast weather conditions at the Port of Turku in Finland, leveraging machine learning to optimize resource allocation and streamline operations.

## Problem Statement

In a team of data scientists for a shipping company based in Turku, Finland, our focus lies in predicting weather conditions at the port. Given that a significant portion of Finland's imports pass through these ports, accurate weather forecasting is crucial. Rainy conditions disrupt logistics and operations, impacting supply chain efficiency. Our objective is to forecast rain 4 hours in advance, facilitating resource optimization and potentially reducing operating costs by up to 20%.

## Task Overview

As data scientists, our responsibility revolves around creating an ML-based solution using historical weather data spanning a decade from the Port of Turku. Our solution aims for continuous learning, enabling the port to optimize its operations effectively.

## Approach

1. **Setting up Resources and Tools**
   - **MLflow:** Leveraging an open-source tool for efficient experiment tracking and management.
   - **Azure Machine Learning:** Utilizing Microsoft Azure to harness the power of cloud services for machine learning.
   - **Azure DevOps:** Employing DevOps practices for streamlined development and deployment workflows.

2. **Data Preprocessing:** Preparing the historical weather data for model training and validation.

3. **Pipeline Construction:** Developing a robust pipeline to facilitate model training, evaluation, and deployment.

4. **Evaluating and Packaging Models:** Assessing model performance and packaging the solution for effective deployment.

This project demonstrates how to use  open-source tools like MLflow with cloud services like Azure, offering insights into solving real-world problems while optimizing operations at the Port of Turku.

we will start by installing MLFLOW and run it using *mlflow ui*
![](https://media.discordapp.net/attachments/1183717517982703667/1184798802352279635/Screenshot_from_2023-12-14_11-06-24.png?ex=658d48de&is=657ad3de&hm=b3d9e811f86afb503e0ec41e8295ef36dec0dcd652928ffbc3bb683553897a59&=&format=webp&quality=lossless&width=1440&height=261)
And then we will create a new resource group named *(Learn_MLOps)* choosing our region as France central and it s the group when we will manage all services related to our ML solution.
![](https://media.discordapp.net/attachments/1183717517982703667/1184800456682242098/Screenshot_from_2023-12-14_11-13-43.png?ex=658d4a69&is=657ad569&hm=52f2022396b7bbaa775d45ebc195dd6331d8480972c6babfc698a23223ee8418&=&format=webp&quality=lossless)

Now ,resource group and after creating our resource group we will creat an azure machine learning workspace,for tracking and managing training, deployment and
monitoring experiments.
After creating this workspace, the platform will deploy all the resources that this service needs
such as: Blob storage, key Vault, application insights. These resources will be consumed or
used via the workspace and the SDK.
![](https://media.discordapp.net/attachments/1183717517982703667/1184805332933939221/Screenshot_from_2023-12-14_11-33-04.png?ex=658d4ef3&is=657ad9f3&hm=f6eaed6f6cf22a3e0c802e01771629bbaf670b878d8e2f7fa70d051e209f03ec&=&format=webp&quality=lossless)


In the next step we will creat an azure devops account ,and create a new project named *“Learn_MLOps”*  Then we go to the repos and we import a repository from github https://github.com/FahdKalloubi1/MLOps_SDAD
![](https://media.discordapp.net/attachments/1183717517982703667/1184807488923639879/Screenshot_from_2023-12-14_11-41-35.png?ex=658d50f6&is=657adbf6&hm=d20a59e6787cd96b9fce31a95735caaea025e77fe5d401f448ab4ae628a21df7&=&format=webp&quality=lossless&width=1440&height=471)

We clone the data and processed it locally , but in our case we will process it and training ml moden and implement it in cloud also , then we will start by create a computing resource in the cloud (Microsoft azure) .
we will create a new computing instance named *"azureml"* with this configurationStandard_E4ds_v4 (4 cores, 32 GB RAM, 150 GB disk)
then we will be able to prepare the data train and deploy the model 
![](https://media.discordapp.net/attachments/1183717517982703667/1184810884518920274/Screenshot_from_2023-12-14_11-55-10.png?ex=658d541f&is=657adf1f&hm=88df4ef3ed45d509dae802bb0ad7bb3dcb1ee115f8757620639d5e6063545a18&=&format=webp&quality=lossless&width=1440&height=461)
we will select the JupyterLab option. JupyterLab is an open source web user
interface. It comes with features like a text editor, code editor, terminal, and custom
components built in an extensible manner.
## Processing and pipeline construction 
after opening JupyterLab ,we will start by cloning our repository from azure devops
using this command :
git clone https://xxxxxx:password@dev.azure.com/xxxxxx/Learn_MLOps/_git/Learn_MLOps
The password is generated after cloning the repo from azure devops and the XXXXXX= "username"
![](https://media.discordapp.net/attachments/1183717517982703667/1184813974521385030/Screenshot_from_2023-12-14_12-07-25.png?ex=658d5700&is=657ae200&hm=3fcda89f7ca033fdefb4c54eecce39b4be1280861221b9a9567116f2f91b0494&=&format=webp&quality=lossless)
Now , we will start our preprocessing so go to preprocessing folder and we run the notebook named "Dataprocessing_register.ipynb"
and wew ill upload to azureml our processed data named 'processed_weather_data_portofTurku'
![](https://media.discordapp.net/attachments/1183717517982703667/1184825273242365992/Screenshot_from_2023-12-14_12-51-46.png?ex=658d6186&is=657aec86&hm=7118c6fb183d605dd4b00137d92bd25b080506773ab0ae4f5fbe414cab5acd91&=&format=webp&quality=lossless&width=1229&height=660)


now after processing our data and save it , we will start train our model.
for that we will open the folder named 'ML_Pipeline'  and run the notebook ML_Pipeline.ipynb

![](https://media.discordapp.net/attachments/1183717517982703667/1184827554750812271/Screenshot_from_2023-12-14_13-01-22.png?ex=658d63a6&is=657aeea6&hm=8d89f338313e56d6aee7f17aecb8ce80847dc8f6604117c07b1e4ef6671e582c&=&format=webp&quality=lossless&width=1281&height=660)
 we will start by importing the moduls that we need , and define our workspace , befoure extracting our data 'processed_weathed_data_prtfTurk from our azureMl data assets
and then we creat some jobs using SVM and RF Machine learning model 
![](https://media.discordapp.net/attachments/1183717517982703667/1184836654205243423/Screenshot_from_2023-12-14_13-36-51.png?ex=658d6c1f&is=657af71f&hm=05f87e68a8c625e2ffa63ccd857c415a53aa97c799374057890119591dd12fe0&=&format=webp&quality=lossless&width=1440&height=436)
and we can also discover our model list then we will find each model with its description  then we can be able to deploy it if we want 
![](https://media.discordapp.net/attachments/1183717517982703667/1184837454574923846/Screenshot_from_2023-12-14_13-39-05.png?ex=658d6cde&is=657af7de&hm=d6e98d0c896c6d9b2d597d8b81640f9e76084cb30e9443789584cfbde16a9467&=&format=webp&quality=lossless&width=1440&height=503)
## Evaluating and packaging our models
This section guides you through evaluating our model's performance and preparing it for deployment. Open the "model_evaluation_packaging.ipynb" notebook in the "Model_evaluation_packaging" folder within JupyterLab and follow these steps:

Connecting and Importing Artifacts

Connect to the ML workspace using Workspace().
Import serialized scaler and model files via the Model() function.
Loading Artifacts for Inference

Read and load the scaler file using pickle.
Utilize the ONNX runtime with InferenceSession() to load the model for predictions.
These steps set the groundwork for assessing our model's performance and packaging it efficiently for deployment

![](https://media.discordapp.net/attachments/1183717517982703667/1184839616247894076/Screenshot_from_2023-12-14_13-49-20.png?ex=658d6ee1&is=657af9e1&hm=ee2eb14308af97a792156ce5796dc3ffdaf45d7566e20e75b770ee8ce6d54ba6&=&format=webp&quality=lossless&width=720&height=417)
# Conclusion 
Conclusion
This repository presents a comprehensive approach to forecasting weather conditions at the Port of Turku in Finland using Azure DevOps, MLflow, and Azure Machine Learning SDK. By addressing the challenges faced by a shipping company operating in this crucial port, we aimed to optimize operations and streamline resource allocation.

Key Sections Covered:
Problem Statement and Task:

Defined the challenge of predicting rainy conditions 4 hours in advance at the port to optimize resource usage and reduce operational costs.
Project Workflow:

Outlined the workflow involving setting up resources and tools, data preprocessing, pipeline construction, and model evaluation and packaging.
Code Execution Challenges:

Encountered issues such as missing modules like Matplotlib and Seaborn, and provided solutions to resolve these dependencies.
Model Evaluation and Packaging:

Detailed steps for connecting to the workspace, importing artifacts, and loading these artifacts for model inference.
Overall Objective:

Through these processes, our aim is to create an ML-based solution that continuously learns from historical weather data to optimize operations at the Port of Turku, ultimately contributing to more efficient supply chain operations.
This repository serves as a comprehensive guide, showcasing how data science methodologies, cloud services, and open-source tools can converge to address real-world challenges, emphasizing the importance of leveraging technology for operational efficiency and cost reduction in the shipping industry.

