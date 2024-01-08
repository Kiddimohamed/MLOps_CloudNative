# MPI_Prediction
## I. Introduction
High-performance computing (HPC) plays a crucial role in tackling complex scientific and engineering challenges, particularly those involving partial differential equations (PDEs). However, HPC simulations that involve PDEs often require extensive communication between the computing nodes, which can slow down the process. In the world of HPC, one of the key tools for communication is the Message Passing Interface (MPI). Optimizing how MPI handles this communication is vital to achieving top-notch performance in PDE simulations.

The challenge, though, lies in the multitude of available MPI algorithms, with the best one depending on various factors like the nature of the problem, mesh size, compiler, problem dimensionality, and communicator characteristics. To overcome this challenge, an innovative approach has emerged: a machine learning-based model designed to predict the optimal MPI algorithm for specific PDE simulations. This model considers several critical parameters related to the simulation, such as the physical problem at hand, mesh size, compiler settings, problem dimensions, and communicator traits

This study underscores the significance of careful selection when it comes to input features and training data, ensuring the model's accuracy and its ability to generalize to a wide range of PDE simulation scenarios. Ultimately, this research represents a significant step forward in the quest for more efficient and effective high-performance computing for PDEs, opening doors to new possibilities in various scientific and engineering domains.
### 1.1 Describe the Data Set
We have a dataset with 23,544 entries and 8 columns. <br>
The columns are as follows: <br>
* MeshSize: An integer column that likely represents the size or granularity of a mesh used in simulations.
* Compiler: A categorical (object) column indicating the type or version of the compiler used in the simulations.
* NumberofVariables: An integer column indicating the number of variables involved in the simulations
* VariableType: A categorical (object) column represents the type of the variables.
* NumberofNodes: An integer column specifying the number of computational nodes used in the simulations.
* NumberofCores: An integer column indicating the number of processor cores used in the simulations.
* Dim: An integer column representing the dimensionality of the simulations.
* Communication: A categorical (object) column .<br>
*-* The objective of this project is to use the first seven columns (MeshSize, Compiler, NumberofVariables, VariableType, NumberofNodes, and NumberofCores,Dim) as features to predict the eight column (Communication)

![Show_num_instant](IMG/Number%20of%20Instances%20per%20Class.jpg)
but abvoisly we need some pre processing before going through our analyse due to a lot of problem ,as the figure above mention there is a balncing data problem and of course some data type and normalization problem 

## II. Preprocessing

### 2.1 Check for Non-null Values
```python
#delete nan values
Data.dropna(axis = 0, how = "any", inplace=True)
#Check the existence of null vaues
Data.isna().sum()
```

| Column             | Number of Null Values |
|--------------------|-----------------------|
| MeshSize           | 0                     |
| Compiler           | 0                     |
| NumberofVariables  | 0                     |
| VariableType       | 0                     |
| NumberofNodes      | 0                     |
| NumberofCores      | 0                     |
| Dim                | 0                     |
| Communication      | 0                     |



### 2.2 Check Data Types
As we noticed in the table bellow , some of our features are strings or objects, so we should convert or encode all the variables to become numerical values.

| Column             | Data Type |
|--------------------|-----------|
| MeshSize           | int64     |
| Compiler           | object   |
| NumberofVariables  | int64     |
| VariableType       | object   |
| NumberofNodes      | int64     |
| NumberofCores      | int64     |
| Dim                | int64     |
| Communication      | object   |

### 2.3 Data Encoding
In our classification project, we leverage the power of the OrdinalEncoder – a valuable tool for transforming categorical data into numerical form. Categorical variables often play a crucial role in machine learning tasks, and the OrdinalEncoder helps us convert these categorical attributes into a structured order that our classification algorithms can interpret effectively. 
<br><br>
This encoder assigns a unique integer to each category, preserving the ordinal relationship between them, which is especially beneficial when dealing with features with inherent ranking or hierarchy. By incorporating the OrdinalEncoder into our project, we enable our classification models to make sense of categorical data and make more accurate predictions
<br><br>
And we talk in this case about The COMPILER , VARIABLETYPE and COMMUNICATION columns , that are already object with string Values 


| Compiler after encoding | Compiler after encoding | VariableType Before encoding | VariableType after encoding | Communication before encoding | Communication after encoding |
|-------------------------|--------------------------|-------------------------------|-----------------------------|-----------------------------|-----------------------------|
| GCC                   | 0                      | int 8                         | 5                         | Persistant\_p2p              | 6                         |
| Intel                 | 1                      | int 16                        | 2                         | Blocking\_Neighbor\_alltoallv | 0                         |
|                       |                        | int 32                        | 3                         | Non\_Blocking\_Neighbor\_alltoallv | 3                    |
|                       |                        | int 64                        | 4                         | Non\_Blocking\_alltoallv      | 4                         |
|                       |                        | float 32                      | 0                         | Non\_Blocking\_p2p           | 5                         |
|                       |                        | float 64                      | 1                         | Blocking\_alltoallv           | 1                         |
|                       |                        |                               |                           |         Blocking\_p2p                                      | 2                         |




### 2.4 Normalize the Dataset
After Converting all our features and encoding all the string value , we will go through another important point, we wil check the mean value of each features.
![Communication_vs_measurements](IMG/Communication%20vs%20measurements.jpg)
We can easily notice that the mean value of each feature is significantly different from the others. For example, the 'meshSize' feature has a mean in the scale of 10^8  instead of the expected 0.5 for the compiler. <br>
<br> This disparity will have an impact on our classification. To address this issue, we will apply a normalization method. In this case, we will use the Min-Max Scaler to transform all the values into a range of 0-1. This will ensure that every feature carries the same importance in terms of power.
<br>
![Communication_vs_measurements_after_normalization](IMG/Communication%20and%20measurements%20after%20Normalization.jpg)
After normalization, we can now apply our technique to this new dataset, as there are no normalization issues to worry about.


## III. Balancing of Data
In the first section, we mentioned that the imbalance in our data would pose significant challenges when training our model. To address this issue, we will apply several balancing techniques, starting with oversampling, SMOTE-NC, and finally, CTGAN. We will provide a detailed explanation of each technique in its respective section. However, before delving into that, we will use the unbalanced data as a reference to ensure that our new data doesn't lose the valuable information it already contains.
![imbalancing data features](IMG/imbalancing_data.png)

Please note that we will exclude the undersampling method, as it involves removing data from the majority class, reducing our dataset from 13,000 to just 128 values in the minority class. This would result in a significant loss of information.

Our focus will now be on these three techniques, and we will compare their effectiveness.
### III.1 Before Balancing
####   III.1.1 Feature Selection (FS)
Feature selection (FS) is a crucial step in any data analysis or machine learning project. It involves the process of identifying and choosing the most relevant features from a dataset while discarding irrelevant or redundant ones. 
<br>
In the first step we will focus in the filter Methods, or The statistical methods As  MI-Score , Chi2 or SelectKBest ..
before going to Feature Importance from Tree-Based Models, because many machine learning models, especially tree-based ones like Random Forest and XGBoost, provide feature importance scores. These scores can be used for FS, with less important features pruned.
#####      III.1.1.1 MI Score

The Mutual Information (MI) score is a measure of the statistical dependence or mutual dependence between two random variables.
We will apply it in Our data between the target and the Features to find the relations between them. <br>
<br>
<br>
The Mutual Information between two random variables X and Y is defined as:
MI(X, Y) = ∑∑ P(x, y) * log(P(x, y) / (P(x) * P(y)))

Where:

- `P(x, y)` is the joint probability distribution of X and Y.
- `P(x)` and `P(y)` are the marginal probability distributions of X and Y, respectively.

In practice, when using Mutual Information for feature selection, you can approximate these probabilities using your dataset. The formula can be simplified as follows:

MI(X, Y) = ∑∑ p(x, y) * log(p(x, y) / (p(x) * p(y)))


Where:

- `p(x, y)` is the empirical joint probability of observing X and Y together in the data.
- `p(x)` and `p(y)` are the empirical marginal probabilities of X and Y, respectively.
<br>

<br>


![Mutual Information Score comarison](IMG/Mutual_Information_Score_Comparison.jpg)
We applied the MI Score to our data; however, unfortunately, the values obtained were very small. This is primarily attributed to the imbalanced nature of our dataset. We anticipate that we will encounter similar low values even after applying each balancing technique
#####      III.1.1.2 CHI-2 Score
The Chi-squared  is a statistical test used to determine whether there is a significant association between two categorical variables( the target and each feature).
![CHI2_Score comarison](IMG/Chi-squared%20Values%20for%20Features%20before%20balancing.png)
A higher Chi-squared value indicates a stronger association or dependence between the feature and the target variable. This means that 'meshSize' and 'compiler' are the most influential features affecting the category. However, it's important to note that we cannot conclusively adopt this information until we compare these values with the other features after balancing the data.

#####      III.1.1.3 K-best Score
The K-best refers to a feature selection technique where you select the top 'k' most important features from your dataset based on some scoring or ranking criterion. The 'k' in "KBest" represents the number of features you want to retain.
it used to reduce dimensionality and improve model performance by focusing on the most informative features while discarding less relevant ones.
<br> <br>
We start by defining a scoring method that quantifies the importance or relevance of each feature with respect to the target variable , and we will adopt two scoring method ,CHi2 and Mutual information (MI_score)
![K_best_CHI2_Score comarison](IMG/Selected_features_name_using_Chi2.png)
![K_best_MI_Score comarison](IMG/Selected_features_name_using_MI.png)
It's evident that the compiler, mesh size, and dimension stand out as key factors in this scenario. However, in order to determine their significance, we must juxtapose these values with those from other sampling methods before identifying the meaningful  features.

####   III.1.2 Feature Importance (FI)
Feature importance is about understanding, evaluating, and selecting the most relevant features in a dataset. 
t refers to the process of determining and quantifying the influence or relevance of each feature in a dataset <br> 
But before going through the result we will start by defining the metrics or the score that we will use in our project 
We begin by assessing the correlation between the target variable and the features, which serves as a fundamental technique in our analysis. Subsequently, we establish a ranking system for each target variable. This ranking is determined by aggregating multiple scores obtained from various feature selection methods, including Recursive Feature Elimination (RFE), the absolute values of parameters in Linear Regression, Ridge Regression, Lasso Regression, and Random Forest (RF).
<br><br>
For each target, the ranking is calculated by considering the mean of these feature importance scores. This approach allows us to identify the most influential features for each specific target, which is essential for predictive modeling and data analysis.

Before delving into the final rankings, let's take a closer look at the individual scoring mechanisms used to assess the importance of features for each target.

#####    III.1.2.1  The correlation Coefficient 
The correlation is typically expressed as a correlation coefficient, which can take values between -1 and 1. The coefficient signifies the direction (positive or negative) and strength of the relationship between variables
In our case we can easily notice that our target depends on the compiler in a positive correlation  meaning that as one variable increases, the other also increases proportionally.
and in other way depends on a negative correlation with Dimension and the number of variable implying that as one variable increases, the other decreases proportionally.
![Correlation](IMG/Correlation_between_features.png)
In this instance, we observe that the compiler, number of nodes, and dimension exhibit a stronger correlation with the target variable. The compiler displays a positive correlation, while the others show a negative correlation.

#####    III.1.2.2  Recursive Feature Elimination (RFE)
RFE begins by assigning an importance score or ranking to each feature in the dataset. This ranking can be based on various criteria, such as coefficients in a linear model, feature importances in a tree-based model, or correlation with the target variable.
#####    III.1.2.3  the absolute values of parameters in Linear Regression
The coefficients represent the weights assigned to each feature in the linear equation used for prediction. Taking the absolute values of these coefficients provides a measure of their magnitude regardless of their direction (positive or negative). This can be useful for understanding the relative importance of features in the model.
<br><br>
For a simple linear regression model with one feature, the equation looks like this:

y = b0 + b1 * x

y is the predicted value (the target variable).
b0 is the intercept (the value of y when x is 0).
b1 is the coefficient associated with the feature x.
In the context of lr.coef_ , it stores the values of b1 for all features in a multiple linear regression model. Each feature has its own coefficient, which indicates its importance and direction (positive or negative) in the linear equation


#####    III.1.2.4  Ridge Regression Coefficient
Ridge Regression is a linear regression technique that adds a regularization term to the standard linear regression cost function.
<br> <br>
In Ridge Regression, the ridge.coef_ attribute provides information about the coefficients (weights) of the features in the trained Ridge Regression model. These coefficients represent the impact or contribution of each feature to the predicted outcome.

#####    III.1.2.4  Lasso Regression Coefficient
Lasso is a linear regression technique with a regularization term that encourages the model to have sparse coefficients ,
it  is similar to Ridge Regression, but it uses a different type of regularization called L1 regularization
<br><br>
In Lasso Regression, the lasso.coef_ attribute provides information about the coefficients (weights) of the features in the trained Lasso Regression model. These coefficients represent the impact or contribution of each feature to the predicted outcome. The distinctive feature of Lasso Regression is that it encourages some of these coefficients to be exactly zero, effectively performing feature selection.

#####    III.1.2.6 Random Forest Coefficient 
Random Forest is an ensemble learning method that builds multiple decision trees during training and combines their predictions to improve overall performance and reduce the risk of overfitting. 
we will explain it more later but now we will focus on the coefficient that we can extract from ot to use it in a features importance 

<br> <br>
It indicates how influential each feature is in making predictions with the Random Forest ensemble. 
The 'feature_importances' variable will contain an array where each element represents the importance score for a feature. Features with higher importance scores are considered more influential in making predictions, while features with lower importance scores have less impact.
<br>
<br>
 And As a result we get this table :

|                   | Lasso | LinReg |   RF  |  RFE  | Ridge |  Mean |
|-------------------|-------|--------|-------|-------|-------|-------|
|    MeshSize       | 0.00  | 0.02   | 1.00  | 0.17  | 0.13  | 0.26  |
|    Compiler       | 1.00  | 0.07   | 0.28  | 0.67  | 0.60  | 0.52  |
| NumberofVariables | 0.81  | 0.06   | 0.19  | 0.50  | 0.51  | 0.41  |
|  VariableType     | 0.15  | 0.00   | 0.29  | 0.00  | 0.00  | 0.09  |
|  NumberofNodes    | 0.00  | 0.96   | 0.00  | 0.83  | 0.75  | 0.51  |
|  NumberofCores    | 0.00  | 1.00   | 0.09  | 1.00  | 1.00  | 0.62  |
|  Dim              | 0.49  | 0.03   | 0.04  | 0.33  | 0.22  | 0.22  |

 ![Mean Of features_importance](IMG/Features_importance_mean.png)

Using our simple data set , we notice that our target 'Communication' is highly depend on Number of cores , and The compiler , but we will also test the other data set 'After balancing' because the value of each one still low.

### III.2  Over Sampling (data_Over)
Starting with the first method, Oversampling. This method duplicates the minority classes to achieve balance. Its principle is simple and basic, but it allows our model to focus more on the minority classes during training.
We refer to our new data as 'data_Over.' Next, we will apply statistical and machine learning methods for feature selection and assessing their importance.
####   III.2.1 Feature Selection (FS)
As in the previous section, we will assess feature importance, beginning with Mutual Information (MI) and Chi-Square (Chi) scores, and also applying the k-best technique.
#####      III.2.1.1 MI Score
we can observe that for over simpling method the mesh size and number of cores and number of node are the higher value in Mesh size. But as we said before we can't extract any information before comparing them with the other value in other data with different sampling method .

![Mutual Information Score comparison_over](IMG/Mutual%20information%20Comparison%20for%20Features%20with%20Over%20Sampling.png)


#####      III.2.1.2 CHI-2 Score
In this section, similarly with MI Score value in the oversampled data, we observe that the 'mishSize' exhibits a higher value in comparison to other features. However, we will reserve judgment until the end to extract meaningful information.
![CHI2_Score comparison_over](IMG/Chi-squared%20Values%20Comparison%20for%20Features%20with%20Over%20Sampling.png)
#####      III.2.1.3 K-best Score

![K_best_CHI2_Score over](IMG/Selected%20Features%20Using%20Chi2%20with%20Over%20Sampling.png)
![K_best_MI_Score over](IMG/Selected%20Features%20Using%20MI%20With%20Over%20Sampling.png)

####   III.2.2 Feature Importance (FI)
#####    III.2.2.1  The correlation Coefficient 
##### TODO 
![Correlation_over](IMG/Correlation_between_features_over_sampling.png)

#####    III.2.2.2 The Mean of the coefficient

![Mean Of features importance](IMG/Features_importance_mean_over.png)

### III.3  SmoteNC Sampling (data_Smote_nc)
####   III.3.1 Feature Selection (FS)
#####      III.3.1.1 MI Score

![Mutual Information Score comparison_smoten](IMG/Mutual%20Information%20Values%20Comparison%20for%20Features%20with%20SmoteNC.png)


#####      III.3.1.2 CHI-2 Score
![CHI2_Score comparison smoten](IMG/Chi-squared%20Values%20Comparison%20for%20Features%20with%20SmoteNC.png)

#####      III.3.1.3 K-best Score
![K_best_CHI2_Score smoten](IMG/Selected%20Features%20Using%20Chi-squared%20with%20SmoteNC.png)
![K_best_MI_Score smoten](IMG/Selected%20Features%20Using%20MI%20with%20SmoteNC.png)
####   III.3.2 Feature Importance (FI)
#####    III.3.2.1  The correlation Coefficient 
![Correlation smoten](IMG/Correlation_between_features_smotenc_sampling.png)

#####  III.3.2.2 The Mean of the coefficient
![Mean Of features_importance_smoten](IMG/Features_importance_mean_smoten.png)

### III.4  CTGAN Balancing (combined_df)
####   III.4.1 Feature Selection (FS)
#####      III.4.1.1 MI Score

![Mutual Information Score comparison_ctgan](IMG/Mutual%20Information%20Values%20Comparison%20for%20Features%20with%20CTGAN.png)

#####      III.4.1.2 CHI-2 Score
![CHI2_Score comparison ctgan](IMG/Chi-squared%20Values%20Comparison%20for%20Features%20with%20CTGAN.png)

#####      III.4.1.3 K-best Score
![K_best_CHI2_Score ctgan](IMG/Selected%20Features%20Using%20Chi-squared%20with%20CTGAN.png)
![K_best_MI_Score ctgan](IMG/Selected%20Features%20Using%20MI%20with%20CTGAN.png)

####   III.4.2 Feature Importance (FI)
#####    III.4.2.1  The correlation Coefficient 
![Correlation ctgan](IMG/Correlation_between_features_ctgan.png)

#####    III.4.2.2 The Mean of the coefficient

 ![Mean Of features_importance_ctgan](IMG/Features_importance_mean_CTGAN.png)
### III.5 Comparison
## Classification
### Metrics
#### Accuracy
Accuracy is a commonly used metric for classification problems. It measures the proportion of correctly classified instances over the total number of instances. While it's simple to understand, accuracy can be misleading when dealing with imbalanced datasets where one class is much larger than the other as our first case before balancing the data.

#### F1 Score
The F1-score is a metric that combines precision and recall. It provides a balance between false positives and false negatives in binary classification problems. It's especially useful when the cost of these errors varies, and you want to find a balance between them.
#### Log Loss (cross-entropy loss)


Log loss, also known as logistic loss or cross-entropy loss, is a metric used to evaluate the performance of a classification model. It measures the performance of a model where the predicted output is a probability value between 0 and 1.

The formula for log loss is:

Log Loss = -1/N * Σ (y_i * log(p_i) + (1 - y_i) * log(1 - p_i))


- \(N\) is the number of samples or instances.
- \(y_i\) is the actual class label (0 or 1) of the \(i\)th sample.
- \(p_i\) is the predicted probability that the \(i\)th sample belongs to class 1.

The log loss penalizes more significant deviations between predicted probabilities and actual outcomes. A lower log loss indicates better performance, where 0 represents a perfect model that perfectly predicts the true probabilities of each instance.

It's commonly used in scenarios involving binary or multiclass classification problems to assess the accuracy of models that generate probabilities as predictions, such as logistic regression, neural networks with softmax outputs, etc.
### Machine Learning Techniques
In our prediction section we will adopt Machine and deep learning model to more understand the link between features, and build a performance model and compare between them .
For the machine learning models we will use :
- KNN
- Support Vector Machines (SVM)
- Decision tree 
- Random forest 
- Adaboost
- Gradient Boosting Classifier
- XGBoost 
- LightGBM
And for The deep Learning models :
- feedforward neural networks
- Convolutional Neural Networks (CNNs)
- CNN+LSTM
- Recurrent Neural Networks (RNNs)
And then we will compare between each models.
#### KNN 
K-nearest neighbors (KNN) is a simple machine learning algorithm used for classification and regression. It predicts based on the majority of neighboring data points. K, the number of neighbors considered, and the distance metric used are crucial factors affecting its performance. It's easy to understand but computationally intensive for large datasets and sensitive to irrelevant features.
The KNN classification approach is based on the hypothesis that each case of the training sample is a random vector from Rn. Each point is described as x =< a1(x), a2(x), a3(x),.., an(x) > where ar(x) corresponds to the value I of the rth attribute. ar(x) can be either a quantitative or a qualitative variable.

In order to determine the class of a target point, each of the k points closest to xq takes a vote. The class of xq corresponds to the majority class.

#### Support Vector Machines (SVM)
In machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised max-margin models with associated learning algorithms that analyze data for classification and regression analysis. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. SVMs can also be used for regression tasks.
#### Decision tree 
Like SVMs, Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks, and even multioutput tasks. They are powerful algorithms, capable of fitting complex datasets.
Decision Trees are also the fundamental components of Random Forests , which are among the most powerful Machine Learning algorithms available
today.
#### Random forest 
a Random Forest is an ensemble of Decision Trees, generally  trained via the bagging method (or sometimes pasting), typically with max_samples  set to the size of the training set. Instead of building a BaggingClassifier and passing it a DecisionTreeClassifier, you can instead use the RandomForestClassifier class, which is more convenient and optimized for Decision Trees


#### Adaboost
Before Explaining The adaboost model , we must start by explain The boosting .
Boosting (originally called hypothesis boosting) refers to any Ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor. There are many boosting methods available, but by far the most popular are AdaBoost and Gradient Boosting.
 Let’s start with AdaBoost:

One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. This results in new predictors focusing more and more on the hard cases. This is the technique used by AdaBoost. For example, when training an AdaBoost classifier, the algorithm first trains a base classifier (such as a Decision Tree) and uses it to make predictions on the training set. The algorithm then increases the relative weight of misclassified training instances. Then it trains a second classifier, using the updated weights, and again makes predictions on the training set, updates the instance weights, and so on

#### Gradient Boosting Classifier
Another very popular boosting algorithm is Gradient Boosting. Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor.


#### XGBoost (extreme Gradient Boosting)
Gradient boosting classifier based on xgboost.

XGBoost is an implementation of the gradient tree boosting algorithm that is widely recognized for its efficiency and predictive accuracy.

Gradient tree boosting trains an ensemble of decision trees by training each tree to predict the prediction error of all previous trees in the ensemble
#### LightGBM (Light Gradient Boosting Machine)
LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel, distributed, and GPU learning.
- Capable of handling large-scale data.

#### Comparison
In our case , we have four type of data , the data before balancing , and the other after balancing with different technic .
for the simple data we get this results :
###### TODO check thi figures titles 
![Simple Ml ](https://media.discordapp.net/attachments/1183717517982703667/1191419782298337420/Metrics_Simple_ML.jpg?ex=65a55f23&is=6592ea23&hm=6c6cd563e5d2b14df0664480ad342e8a3d0720d2c5b43d5aa4bb0244ba561fda&=&format=webp&width=1140&height=660)
As mentioned earlier, we will focus on three metrics: Log Loss, Accuracy, and F1-Score.

Let's begin with accuracy, where we observe that XGBC and LGBC exhibit the highest values, scoring 0.97 for accuracy and 0.78 for F1-Score. However, both Adaboost and DT appear to be poor choices based on their log loss. The higher the value, the worse the prediction seems to be.

Since the values are still not satisfactory, we'll consider them merely as references to identify which type of balancing might be problematic. In other words, if our model is tested with new data and the accuracy decreases, it suggests that this particular type of balancing complicates the data and muddles the distinction between classes. If not, it means our new data is more meaningful and can be used to create a robust model for predicting Communication Types.

However, we can start with the hypothesis that XGBC and LGBC are the best options for our case.

Now we will explore the other datasets starting by The Over_data (data that we apply the over sampling as a balancing technic)
![Over Ml ](https://media.discordapp.net/attachments/1183717517982703667/1191419856684326963/Metrics_Over_ML.jpg?ex=65a55f35&is=6592ea35&hm=5bb02a82d5b51c4c0f3afea70a388ad97320a5d4a7ce6096985e8113230810ec&=&format=webp&width=1140&height=660)
The first thing we can notice is that the model has become more powerful; the models have become more performant. Then we can discuss the value because the balancing method makes our data richer, and the loss becomes lower.

As for the models, we can notice that KNN and Adaboost have the highest values in accuracy and F1 Score, but their performance is still low because of the highest value of the log loss. On the other hand, RF, XGBC, and LGBC are still the preferable models because of their highest accuracy and F1-score metrics, and the lowest Log Loss.

Also, in this case, we will adopt RF (Accuracy: 0.93, F1-Score: 0.95, cross-entropy loss: 0.21) and XGBC (Accuracy: 0.91, F1-Score: 0.91, cross-entropy loss: 0.23) as the best models for this type of balancing.


Now, we will proceed to the next balancing method, SMOTE-NC, and assess its suitability as an effective balancing technique. To do so, let's refer to the figure below
![Smotenc ML](https://media.discordapp.net/attachments/1183717517982703667/1191419781493031033/Metrics_Smotenc_ML.jpg?ex=65a55f23&is=6592ea23&hm=68c10b87186dad4ac136e5e74e9a99c44c4839acddb2c151c55f3f0dd53b3cbb&=&format=webp&width=1140&height=660)
For this case, we can assume that the model has become more powerful. Therefore, adopting the balancing method is a good approach. Since the accuracy values are closer to each other, hovering around 0.93, except for the elimination of the SVM model, we will rely on the Log Loss to choose the best model. Hence, we can easily consider XGBoost, LightGBM, and Random Forest as the best models. These results align with the two previous outcomes using different datasets.

In this scenario, we will opt for XGBoost (Accuracy: 0.93, F1-Score: 0.93, cross-entropy loss: 0.19), Random Forest (Accuracy: 0.93, F1-Score: 0.94, cross-entropy loss: 0.34), and LightGBM (Accuracy: 0.91, F1-Score: 0.91, cross-entropy loss: 0.32).

Finally we will discuss the CTGAN results :
![CtganMl ](https://media.discordapp.net/attachments/1183717517982703667/1191419857548353586/Metrics_Ctgan_ML.jpg?ex=65a55f35&is=6592ea35&hm=7f569ab87391219db015f841c058165455e1209a3803a08152de755c456642cd&=&format=webp&width=720&height=417)



### The Deep Learning Techniques
Deep learning is a subset of machine learning that involves training neural networks with multiple layers to learn representations of data. It's inspired by the structure and function of the human brain's interconnected neurons. These neural networks consist of layers of interconnected nodes (neurons) that process information, where each layer extracts increasingly complex features from the input data. 

#### feedforward neural networks
 These are the simplest form of neural networks where information travels in one direction—from input nodes through hidden nodes (if any) to output nodes. There are no loops or cycles in the network, and they are commonly used for tasks like classification and regression.
#### Convolutional Neural Networks (CNNs)
Specifically designed for processing structured grid-like data, like images. CNNs use convolutional layers to systematically apply filters to input data, enabling the network to learn features and patterns hierarchically. They've been incredibly successful in image recognition, object detection, and classification tasks.

#### CNN+LSTM
This is a combination of Convolutional Neural Networks and Long Short-Term Memory networks. It's particularly useful for tasks that involve both spatial and temporal features, such as video classification or action recognition in videos. The CNN extracts spatial features, which are then fed into the LSTM to model temporal dependencies.
#### Recurrent Neural Networks (RNNs)
These are neural networks designed to work with sequential data by maintaining a form of memory or context. RNNs have connections that form a directed cycle, allowing information to persist. However, they suffer from the vanishing or exploding gradient problem, which affects their ability to learn long-range dependencies.
#### Comparison

After comparing between the ML models , we will start the second section where we will compare the deep learning models.
![Simple DL ](https://media.discordapp.net/attachments/1183717517982703667/1191419782550012004/Metrics_Simple_DL.jpg?ex=65a55f23&is=6592ea23&hm=a1292156a3e0d0f10d1ec5d52b3a33c7b3d8541f36afce32717734efbc291d5c&=&format=webp&width=720&height=417)
and as before we will use the non balancing data models results as a reference , to know which balancing type is good .

Starting by models with data_over 
![Over Dl ](https://media.discordapp.net/attachments/1183717517982703667/1191419857221210132/Metrics_Over.jpg?ex=65a55f35&is=6592ea35&hm=686ee723384e5a99b9259abd57d66fa00cee0eba1ca9e256f04de2690b37ad2d&=&format=webp&width=720&height=417)


![Smotenc DL ](https://media.discordapp.net/attachments/1183717517982703667/1191419781765672990/Metrics_Smotenc_DL.jpg?ex=65a55f23&is=6592ea23&hm=74ef23927b3d8649baf23487fbf7feff8e2974b9f284dd7d741b88c324ac4161&=&format=webp&width=720&height=417)


![Over Dl ](https://media.discordapp.net/attachments/1183717517982703667/1191419857221210132/Metrics_Over.jpg?ex=65a55f35&is=6592ea35&hm=686ee723384e5a99b9259abd57d66fa00cee0eba1ca9e256f04de2690b37ad2d&=&format=webp&width=720&height=417)
![Ctgan DL ](https://media.discordapp.net/attachments/1183717517982703667/1191419858219454555/Metrics_Ctgan.jpg?ex=65a55f35&is=6592ea35&hm=3d9d6bf3e1b1bd94109bebb25be9fb07b6d251fb339b0e86b11d41fe406f6924&=&format=webp&width=720&height=417)
###  Conclusion


https://en.wikipedia.org/wiki/Support_vector_machine
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
https://docs.getml.com/latest/api/getml.predictors.XGBoostClassifier.html
https://lightgbm.readthedocs.io/en/stable/