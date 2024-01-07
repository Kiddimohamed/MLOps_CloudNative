# MLFlow with SCikitLearn / TensorFlow 20.0(Keras) / Pyspark / DataBricks .


In this lab we'll embark on an exciting journey exploring MLflow in conjunction with various powerful tools and frameworks while working with the MNIST and credit card datasets. In this dynamic environment, we'll delve into the realms of Scikit-Learn, TensorFlow 2.0 (including Keras), PySpark, and DataBricks, leveraging these tools to analyze and model these distinct datasets.

Our focus will extend beyond mere exploration; we'll dive into the practical applications of these technologies. From model development to deployment, our investigations will encompass local model serving and deployment strategies, enabling us to grasp the full spectrum of machine learning lifecycle management.

The inclusion of the MNIST and credit card datasets offers diverse avenues for exploration, allowing us to apply these frameworks to distinct data types and problem domains. Through these datasets, we'll witness firsthand how MLflow enhances the management and tracking of experiments, contributing to streamlined workflows and reproducibility.

This collaborative lab environment provides us with the opportunity to dive into the nuances of each framework while leveraging MLflow to streamline experimentation, reproducibility, and deployment. Get ready to engage in hands-on experiences and uncover the potential synergies between MLflow and these powerful tools, shaping a deeper understanding of their functionalities and capabilities in diverse data contexts.

## MLFlow-scikitlearn 
First of all we will open the notebook  1. MLFlow-Scikitlean.ipynb
In this lab we will use the credit card dataset for fraud detection: https://www.kaggle.com/mlg-ulb/creditcardfraud
### preprocessing
Firstly we started by preprocessing our data ,by dropping the useless column ,and balancing our data , and then devide it in two subset , the normal and anomaly , then we find that the shape for normal class is  (142158, 30) and for the anomaly is (492, 30)
Then we split it to train test validate subsets and of course we finish by using standardscaler to scale our data .



### MLFLow
Now we will train and evaluate our model using MlFlow .
then we will start by tracking_uri by the following command  'mlflow.set_tracking_uri("http://localhost:1234")' then we will access later to the web interface using the following command 'mlflow ui -p 1234'
we will set the experiment name, defining our model, and starting the Mlflow run.
and finally we will save our model using 'mlflow.sklearn.log_model(sk_model, "log_reg_model")'

In the training face , we will use the LogisticRegression and the SVC Model , and we plot the confusion matrix , but our goal is to compare both models using ml flow.
![MLFLow comparaison](https://media.discordapp.net/attachments/1183717517982703667/1192851473223135404/Screenshot_from_2024-01-05_16-25-39.png?ex=65aa9481&is=65981f81&hm=97674a1064c4ae28ed4866a8180311464894af8b544d461a78b61046391fe4b6&=&format=webp&quality=lossless&width=720&height=423)

There are two experiments: the default one and the newly created 'scikit_learn_experiment.' You'll find all the logged parameters and metrics visible. Just continue scrolling downward, and you'll be able to view all the logged artifacts.
and now we will load and logged the model.

loaded_model = mlflow.sklearn.load_model("runs:/5aaae41666ce4ccc8d8793a96001aed6/log_reg_model") 

and the final task is Model validation :parameter tuning with MLflow - Broad search 

The hyperparameter we aim to adjust is how much you want to weigh anomalies compared to normal data points. By default, both are weighted equally. Let's define a list of weights to iterate over:anomaly_weights = [1, 5, 10, 15] 
![](https://media.discordapp.net/attachments/1191490101247758479/1192859017236717671/Screenshot_from_2024-01-05_16-54-10.png?ex=65aa9b87&is=65982687&hm=4f8e50c25fd324dbd80479de26202d6a6a22b119e11dcc96872eceb448ebc831&=&format=webp&quality=lossless&width=720&height=334)
![](https://media.discordapp.net/attachments/1183717517982703667/1192865743713349662/Screenshot_from_2024-01-05_17-22-09.png?ex=65aaa1cb&is=65982ccb&hm=81aa6882c09ef478a32161c6251d2e48ef5fe2ea6d045e42de54ba1a22841735&=&format=webp&quality=lossless&width=720&height=334)

The best overall performances were achieved with anomaly weights of 5 and 10, but it seems that the trend is decreasing as we increase the anomaly weight.


## MLFlow-TensorFlow 2.0 (Keras) 
In this section we will use the MNIST Data set.
Also note that the labels are all integer numbers ranging from 0 to 9, each associated with an image depicting a handwritten digit from 0 to 9.

Since 2D convolutional layers in TensorFlow/Keras expect input data in four dimensions in the format (m, h, w, c), where m represents the number of samples in the dataset, h and w represent the height and width respectively, and c represents the number of channels (three if it's an RGB color image, for instance), you need to reshape your data to adhere to these specifications. Your images are all in black and white, technically having only 1 channel. Therefore, you need to reshape them.

we run our model , Let's now open the MLFlow user interface and check our run.

![](https://media.discordapp.net/attachments/1183717517982703667/1192881248482373745/Screenshot_from_2024-01-05_18-23-56.png?ex=65aab03c&is=65983b3c&hm=80a3df8512886b71858804a760eab4e369a6dbe8eabf8e51698b1d9cb494c11d&=&format=webp&quality=lossless&width=720&height=334)
then we can usually apply the same steps that we are apply in the first section .

