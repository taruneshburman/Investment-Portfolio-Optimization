# Investment-Portfolio-Optimization
Financial accountability Using Machine Learning

Contents
1.Introduction	2
1.1 Use of Machine Learning in Financial accountability	2
1.2 What is Machine Learning	2
2. Problem statement	3
3. Solution 1/3*	4
3.1 Solution 2/3*	6
3.2 Solution 3/3*	7
4. Explaination	8
5. Conclusion	9

















1.Introduction
The term "financial accountability" refers to the obligation for people or organisations to handle their financial resources in an ethical and open way. It entails handling, managing, and reporting financial transactions appropriately as well as making sure that the financial records are correct, comprehensive, and open. Financial accountability is crucial for ensuring that funds are used effectively and efficiently. It can apply to individuals, businesses, non-profit organisations, and government agencies. Also, it aids in avoiding corruption, bad management, and fraud. The creation of proper financial rules and procedures, as well as efficient internal controls to stop financial irregularities, are requirements for financial responsibility. It also entails routine financial reporting to regulators and stakeholders, both internally and internationally. Ultimately, sustaining stakeholder trust and confidence as well as ensuring that resources are being spent in the organization's and the clients' best interests depend on financial accountability.
1.1 Use of Machine Learning in Financial accountability
Financial accountability can benefit from the use of machine learning (ML) to automate procedures, find fraud, and enhance decision-making. These are some examples of ML applications:
1.	Fraud detection: ML algorithms can examine a lot of financial data to find anomalies and strange patterns that might be a sign of fraud. In order to spot recognised fraud trends and spot brand-new ones in real-time, machine learning models can learn from prior data.
2.	Risk evaluation: ML models can be used to evaluate the risk involved in investments or financial transactions. Based on historical data, these models can forecast the possibility of default or other financial hazards.
3.	Consumer segmentation: ML algorithms can examine consumer buying habits and behaviour to divide customers into groups based on their profitability or risk tolerance. This can assist financial organisations in customising their goods and services to better meet the needs of their clients.
4.	Analytics that forecast future financial results based on historical data is known as predictive analytics. These models, for instance, can forecast stock prices, loan defaults, and customer attrition.
5.	Monitoring transactions for potential compliance violations in real-time is possible with machine learning (ML) models. Financial institutions may be able to avoid fines and penalties by doing this.
ML may generally assist financial organisations in automating operations, lowering operating expenses, and increasing the accuracy of financial accountability. To ensure openness and fairness, it's crucial to remember that the use of ML in financial accountability necessitates careful evaluation of the ethical and legal consequences.



1.2 What is Machine Learning
A subfield of artificial intelligence (AI) called machine learning (ML) is concerned with developing statistical models and algorithms that enable computers to learn from data and perform jobs more efficiently. In conventional programming, the programmer creates a set of detailed instructions for the computer to follow in order to do a certain task. With machine learning, the task is carried out by the computer without being explicitly programmed by using what it has learned from the data.
The three main categories of ML algorithms are reinforcement learning, unsupervised learning, and supervised learning :
1.	Supervised learning : The algorithm is trained using labelled data, meaning the input data has predetermined output values, in supervised learning. By identifying patterns and relationships in the data, the algorithm learns to map input data to the appropriate output.
2.	Unsupervised Learning :  The algorithm is trained on unlabeled data in this sort of machine learning, which implies the input data does not contain predetermined output values. Without any privious knowledge of what the desired output should be, the algorithm learns to identify patterns and relationships in the data.
3.	Reinforcement learning : By trial and error and interaction with the environment, the algorithm learns new skills in this type of machine learning. By obtaining feedback in the form of rewards or penalties for its actions, the algorithm learns to engage in behaviours that maximise rewards.
Several industries, including banking, healthcare, marketing, and robotics are among those where machine learning (ML) has multiple uses. Processes could be automated, decision-making could be improved, and new insights could be gained that weren't attainable using more conventional techniques.
2. Problem statement
The issue entails categorising credit ratings and determining, using financial and accounting indicators, whether a company is investment grade or not. The data collection includes 1700 observations of 26 attributes for a group of companies across many industries, including credit rating, investment grade, and other financial and accounting measures. Developing a machine learning model that can accurately classify a company's credit rating into one of 16 categories and determine whether or not it is investment grade is the aim. To do this, the dataset must be split into a training set and a test set in an 80 - 20 ratio.
Using linear regression with Ridge (L1) and Lasso (L2) regularisation is one method for resolving this issue. These methods include a penalty term that is added to the cost function to help prevent overfitting in a regression model.
In contrast to the Lasso regularisation, which adds a punishment term equal to the absolute value of the coefficients' magnitude, the Ridge regularization's penalty term is equal to the square of the coefficients' magnitude. The severity of the penalty term is controlled by the regularisation parameter lambda, which can be adjusted to enhance the effectiveness of the model.
Essentially, the challenge is to develop a model utilising machine learning methods that can distinguish between investment grade and credit rating based on financial and accounting criteria, while simultaneously avoiding overfitting by using Ridge and Lasso regularisation.


3. Solution 1/3*
The following actions can be made to develop a linear regression technique with Ridge (L1) and Lasso (L2) regularisation to forecast if a corporation is in an investment grade or not:
1.	Data loading : Fill a pandas data frame with the information from the CSV file.
2.	Data cleaning : Eliminate any unnecessary columns, and deal with any missing data by either eliminating it or imputing the values that it is lacking.
3.	Data pre-processing : Categorical variables are converted into numerical variables via one-hot encoding, and the numerical variables are normalised by scaling or normalisation.
4.	Data splitting : Use the scikit-learn train-test-split method to divide the data into training and test sets in an 80:20 ratio.
5.	Model training : Using the training data, train the linear regression model with Ridge (L1) and Lasso (L2) regularisation.
6.	Evaluation of the model : Determine the accuracy, precision, recall, F1 score, and confusion matrix to assess the model's performance on the test data.
7.	Parameter tuning : Adjust the regularisation parameter lambda to enhance the model's performance using grid search and cross-validation.
8.	Model deployment : Use the model to be deployed to make predictions based on new data.
Overall, employing a linear regression strategy with Ridge (L1) and Lasso (L2) regularisation can assist in enhancing the precision of determining whether a firm is in an investment grade while also avoiding overfitting.
code

#importing all the necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load the dataset into a pandas DataFrame
dff = pd.read_csv('MLF_GP1_CreditScore.csv')
#printing the dataset
print(dff)
#showing top 5 rows
dff.head()



A_train, A_test, B_train, B_test = train_test_split(dff.iloc[:,:-2],
                                                    dff.iloc[:,-2:],
                                                    test_size=0.2, 
                                                    random_state=42)
A_train_metrics = A_train.iloc[:,:-1]
A_test_metrics = A_test.iloc[:,:-1]
B_train_ig = B_train.iloc[:,0]
B_test_ig = B_test.iloc[:,0]
scaler = StandardScaler()
A_train_metrics_scaled = scaler.fit_transform(A_train_metrics)
A_test_metrics_scaled = scaler.transform(A_test_metrics)
ridge_model = Ridge(alpha=1.0)
# Training the models using the fit() function on the training set
ridge_model.fit(A_train_metrics_scaled, B_train_ig)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(A_train_metrics_scaled, B_train_ig)
B_pred_ig_ridge = ridge_model.predict(A_test_metrics_scaled)
B_pred_ig_lasso = lasso_model.predict(A_test_metrics_scaled)
from sklearn.metrics import precision_score, recall_score, f1_score




# Evaluate the performance of the models using metrics such as accuracy, precision, recall, and F1 score
print('Ridge Model:')
print('Accuracy:', accuracy_score(B_test_ig, B_pred_ig_ridge.round()))
print('Precision:', precision_score(B_test_ig, B_pred_ig_ridge.round(), average='weighted'))
print('Recall:', recall_score(B_test_ig, B_pred_ig_ridge.round(), average= 'weighted'))
print('F1 Score:', f1_score(B_test_ig, B_pred_ig_ridge.round(), average = 'weighted'))

print('Lasso Model:')
print('Accuracy:', accuracy_score(B_test_ig, B_pred_ig_lasso.round()))
print('Precision:', precision_score(B_test_ig, B_pred_ig_lasso.round()))
print('Recall:', recall_score(B_test_ig, B_pred_ig_lasso.round()))
print('F1 Score:', f1_score(B_test_ig, B_pred_ig_lasso.round()))
output
Ridge Model:
Accuracy: 0.7647058823529411
Precision: 0.78781006378455
Recall: 0.7647058823529411
F1 Score: 0.6830793856003939
Lasso Model:
Accuracy: 0.7529411764705882
Precision: 0.7529411764705882
Recall: 1.0
F1 Score: 0.8590604026845637

3.1 Solution 2/3*
Now It was asked to create a logistic regression method using regularisation with Ridge (or L1) and Lasso (or L2) to determine if a corporation is investment grade or not.
Code
#spliting the data into 80 : 20 ratio
A_train, A_test, B_train, B_test = train_test_split(A,B, test_size=0.2, random_state=42)
B_train = np.ravel(B_train)
B_test = np.ravel(B_test)
#Training logistic regression model with Ridge
ridge = LogisticRegression(penalty='l2',solver='liblinear' ,C=1.0)
ridge.fit(A_train,B_train)

#Evaluating the Ridge on the test data
B_pred = ridge.predict(A_test)
accuracy = accuracy_score(B_test, B_pred)
precision = precision_score(B_test, B_pred, average='weighted')
recall = recall_score(B_test, B_pred, average='weighted')
f1 = f1_score(B_test, B_pred, average='weighted')
print("Ridge regularization:")
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1}")
#Training a logistic regression model with lasso on training data
lasso= LogisticRegression(penalty='l1', C=1, solver='liblinear')
lasso.fit(A_train, B_train)
#Evaluating the lasso regularization model on the test data
B_pred = lasso.predict(A_test)
accuracy = accuracy_score(B_test, B_pred)
precision = precision_score(B_test, B_pred, average='weighted')
recall = recall_score(B_test, B_pred, average='weighted')
f1 = f1_score(B_test, B_pred, average='weighted')
print("Lasso regularization:")
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1}")
output
Ridge regularization:
Accuracy: 0.7676470588235295, Precision: 0.7586886205159046, Recall: 0.7676470588235295, F1 score: 0.6928718477944484
Lasso regularization:
Accuracy: 0.7647058823529411, Precision: 0.7501960784313726, Recall: 0.7647058823529411, F1 score: 0.6869465671680044





3.2 Solution 3/3*
Now it was asked to create a method based on neural networks that divides the firm's rating into one of the rating categories and forecasts whether it is investment grade
Code
#Scaling the data
scaler = StandardScaler()
A_train_std = scaler.fit_transform(A_train)
A_test_std = scaler.transform(A_test)
#setting up the Multi Layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes = (100,50), activation='relu',  solver='adam', alpha=0.0001, 
                    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200, 
                    shuffle=True, random_state=42, tol=0.0001, verbose=False, early_stopping=True)
#training the MLP
mlp.fit(A_train_std, B_train)
#predicting the results of the test set
B_pred = mlp.predict(A_test_std)
#evaluating the performance of the classifier on the test set
accuracy = accuracy_score(B_test,B_pred)
precision = precision_score(B_test, B_pred, average='weighted')
recall = recall_score(B_test, B_pred, average= 'weighted')
f1 = f1_score(B_test, B_pred, average='weighted')
#returning the accuracy report
print(f"Accuracy: {accuracy}, precision: {precision}, Recall: {recall}, f1_score: {f1}".format(accuracy, precision, recall,f1))
output
Accuracy: 0.7529411764705882, precision: 0.7010960906101571, Recall: 0.7529411764705882, f1_score: 0.6871158151842051



4. Explaination
The first solution is to estimate a company's investment grade based on its financial and accounting indicators using linear regression with Ridge and Lasso regularisation. Metrics including accuracy, precision, recall, and F1 score are used to assess the models' performance.
The second approach uses logistic regression with Ridge and Lasso regularisation to ascertain whether or not a company is investment grade. Using criteria like accuracy, precision, recall, and F1 score, the models' performance is assessed.
Generally, regularisation approaches are being used by both solutions to reduce overfitting and enhance model generalisation. The type of data and the nature of the problem will determine whether to use logistic regression or linear regression. On the basis of fresh data, make predictions using the deployed model.
The thierd code uses a neural network-based classification method to categorise the firm's credit rating and determine whether or not it is investment grade. The following actions are taken by the code:
1.	The features of the data are first scaled using StandardScaler.
2.	2. To create a Multi-Layer Perceptron (MLP) classifier, the scikit-learn library's MLPClassifier class is utilised. The MLP has two hidden layers, each with 100 and 50 neurons, and employs the rectified linear unit (ReLU) activation function. The weights are optimised using the Adam optimizer, with alpha, the regularisation parameter, set to 0.0001. There are additional settings for batch size, learning rate, and maximum iterations.
3.	3. The scaled training data are then used to train the MLP classifier using the fit approach.
4.	The predict method is then applied to the trained classifier to forecast the labels for the test data.
5.	The performance of the classifier is evaluated using the Scikit-Accuracy Score, Learn's Precision Score, Recall Score, and F1 Score algorithms. Thus, these standards are used to determine the classifier's accuracy, precision, recall, and F1-score.
6.	The print command is then used to print the performance metrics in the console.
The output displays the classifier's accuracy, precision, recall, and F1-score on the test set. The accuracy is 75.29%, the precision is 70.11%, the recall is 75.29%, and the F1-score is 68.71%.
5. Conclusion
The three proposed techniques are designed to forecast a company's investment grade status using financial and accounting data. The first two solutions apply, respectively, Ridge and Lasso regularisation to logistic regression and linear regression. The third option makes use of the MLPClassifier class from the scikit-learn library and a neural network-based classification technique.
All three strategies employ regularisation to reduce overfitting and improve model generalisation. Regression techniques are selected based on the nature of the issue, with logistic regression being used for binary target variables and linear regression being used for continuous objective variables. The third approach uses a neural network-based classification algorithm to attain accuracy, precision, recall, and F1-score values of 75.29%, 70.11%, and 68.71%, respectively.
Overall, these solutions offer various methods for determining an organization's investment grade based on financial and accounting characteristics, with the neural network-based categorization method outperforming the others. However, the type of the data and the precise requirements of the application will determine which method is best appropriate for a given scenario.
