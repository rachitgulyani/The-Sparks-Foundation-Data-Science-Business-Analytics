#IMPORTING THE REQUIRED LIBRARIES
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
print("Data is impoting...")
data=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
print("Data has been imported")
X=data.iloc[:,0:1].values
y=data.iloc[:,1:2].values

#VISUALISING THE DATASET
plt.scatter(X,y)

#SPLITTING THE DATASET IN TRAIN AND TEST SETS
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#TRAINING THE DATASET
print("Training dataset...")
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#PREDICTING THE RESULTS OF THE DATASETS
print("Predicting the results...")
y_pred=regressor.predict(X_test)

#VISUALISING THE TRAIN SET RESULTS
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='red')

#COMAPRING THE ORIGINAL AND PREDICTED VALUES
df = pd.DataFrame({'Actual': y_test[:,0], 'Predicted': y_pred[:,0]}) 
print(df)

#PREDICTING THE SCORE IF SOMEONE STUDIES FOR 9.5 HRS
Own_predicted_value=regressor.predict([[9.5]])
print("The predicted value for 9.5 hrs is",Own_predicted_value)


