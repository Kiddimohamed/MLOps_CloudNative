# ML with Amazon SageMaker
Welcome to this lab! Here, we delve into Amazon SageMaker, an impactful cloud-native tool for training and deploying ML models in production. Our focus: 1. Data processing using SageMaker. 2. Training models via SageMaker SDK with built-in algorithms, covering data prep, job configuration, running, and deployment. 

Our task: Train an XGBoost model, fine-tuning its hyperparameters. 

## Problem description:
Find the best strategies to improve for the next marketing campaign. How can the financial
institution (a bank) have greater effectiveness for future marketing campaigns? To answer it,
we need to analyze the latest marketing campaign carried out by the bank and identify
patterns that will help us draw conclusions in order to develop future strategies.
## 1. Data processing using SageMaker
But before going through our task , we will start by preparing our environment  
starting by Launch an Amazon SageMaker notebook instance
![Starting the notebook](https://media.discordapp.net/attachments/1183717517982703667/1183718206477709322/Screenshot_from_2023-12-11_11-33-02.png?ex=65895a7c&is=6576e57c&hm=e653170a3530aae7038ff6e584b8b460e0c0aca129ce12541ebbe34a88ab3594&=&format=webp&quality=lossless&width=720&height=312)
 
and then we will go to start by downloading and extracting our data 

![extract and load data ](https://media.discordapp.net/attachments/1183717517982703667/1183720934415282196/Screenshot_from_2023-12-11_11-43-40.png?ex=65895d07&is=6576e807&hm=2d19d3ea4d28a99df6bcfd9ac9b6eacf62fe6e7690fdb3010767528ea8cbdfb4&=&format=webp&quality=lossless&width=720&height=312)
 you will find it in a folder name it *bank-aditional* 

next, we will Uploading the dataset to Amazon S3: We'll utilize the default 'Bucket' provided by SageMaker in our region. In our notebook, using the SKLearnProcessor object from the SageMaker SDK, we configure the processing task by specifying the scikit-learn version and infrastructure (e.g., ml.m5.xlarge instance). Finally, we run the Job, providing the script name and dataset paths in S3 to the SageMaker processing environment

![extract and load data ](https://media.discordapp.net/attachments/1183717517982703667/1183717845981466684/Screenshot_from_2023-12-11_09-09-56.png?ex=65895a26&is=6576e526&hm=e0e9508fa8086cd934a1b768f64c275f87068d78e3804e04fca120cb431aae4b&=&format=webp&quality=lossless&width=720&height=587)

In a terminal, we can use the AWS CLI to retrieve the processed training and test
sets located at the previous path. 

To do this, go to the “Launcher” tab and click on
“Terminal” and then run the following two commands to copy the files from S3 to the local machine as well as two other commands to view the first records

![extract and load data ](https://media.discordapp.net/attachments/1183717517982703667/1183717846220546110/Screenshot_from_2023-12-11_09-12-58.png?ex=65895a26&is=6576e526&hm=45ad8987397768e8bb4921ce6b2e10291b41f79096c86624f03ec3b25cb27da6&=&format=webp&quality=lossless&width=720&height=131)

## 2. Training a model using SageMaker SDK with built-in algorithms
Having explored dataset preprocessing in Amazon SageMaker, now we'll proceed to train XGBoost model using the Boston Housing dataset.(https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset).
### 2.1 Data preparation
Open the notebook named XGboostmodel, 
Firstly , we will load our data , and transform its form due  to the official sagemaker documentation where they require that Csv file should not contain n the header record and that the target variable must be at the beginning.
Let’s divide our dataset into 90% training and 10% testing and save these two partitions in two csv files without headers and without indices.


Finaly we upload these two files to S3 in two separate folders. We will use the
default “Bucket” bucket created by SageMaker in the region where we are running our
notebook.
![](https://media.discordapp.net/attachments/1183717517982703667/1184068618791690280/Screenshot_from_2023-12-12_10-42-04.png?ex=658aa0d5&is=65782bd5&hm=34be73b6a36d33dee5ff7cceb707e732ee78e1f8795eed813fb5973be7174d9c&=&format=webp&quality=lossless&width=720&height=196)

And then we train Our XGBoostmodel but before wew ill define the estimatore .
The Estimator object from *'sagemaker.estimator.Estimator'* forms the backbone of model training within SageMaker. It serves as the gateway to selecting the right algorithm and specifying training infrastructure requirements.
SageMaker's algorithms are encapsulated in Docker containers. Leveraging boto3 and the image_uris.retrieve() API, we effortlessly identify the name of the XGboost algorithm based on the region where our job is executed. 
then we will configure our training job and we need to define the hyper parameters and the last part before running our job is define data channels: A channel is a named source of data passed to a  SageMaker estimator. All built-in algorithms need at least one training channel, and  many also support additional channels for validation and testing. We will define two  channels: one for training and the other for validation

![](https://media.discordapp.net/attachments/1183717517982703667/1184077783115243560/Screenshot_from_2023-12-12_11-22-02.png?ex=658aa95e&is=6578345e&hm=cfbd0c04373cb15933c063f16a3092e781c206dec8e5ae57c850ee25192a8b29&=&format=webp&quality=lossless&width=720&height=225)


we can notice that our training job  then we will go to deloy our model 
We deploy the model using the deploy() API. As this is a test endpoint,

![](https://media.discordapp.net/attachments/1183717517982703667/1184079015905075210/Screenshot_from_2023-12-12_11-26-55.png?ex=658aaa84&is=65783584&hm=fa36eae1847007c6265e98af964a825a787287fd75a8854538b5863e226614d4&=&format=webp&quality=lossless&width=720&height=256)

We can also predict the output of several samples and We can also predict using the endpoint directly using the “invoke_endpoint()”
function of the “runtime” object.
Finally, and to avoid any additional charges, we can delete the endpoint

![](https://media.discordapp.net/attachments/1183717517982703667/1184076975892090900/Screenshot_from_2023-12-12_11-18-31.png?ex=658aa89e&is=6578339e&hm=ba5b86a756dc41c7c4f11e61be4a4c3ee33da2b25076678a2469c8d5192d6707&=&format=webp&quality=lossless&width=720&height=515)

# Conclusion 
In conclusion, this lab journeyed through the dynamic landscape of Amazon SageMaker, empowering users to explore machine learning from data preprocessing to model training and deployment. Leveraging the Estimator object and a range of SageMaker's functionalities, participants delved into algorithm selection, infrastructure configuration, and Docker containerization of SageMaker's robust algorithms. As a comprehensive platform, SageMaker continues to serve as an invaluable tool, bridging the gap between experimentation and production-grade machine learning models for diverse applications
