# Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by reading the CSV file, encoding categorical variables (Position), and preparing feature (x) and target (y) variables.
2. Split the data into training and testing sets using train_test_split.
3. Train a Decision Tree Regressor model on the training data and make predictions on the test set.
4. Evaluate the model's performance using metrics like Mean Squared Error, Mean Absolute Error, and R² Score, and make a salary prediction for a given input.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Prajin S
RegisterNumber:  212223230151
*/
```
```Python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv('Salary.csv')
df.head()
df.info()
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()
x=df[['Position','Level']]
y=df['Salary']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
ypred=dt.predict(xtest)
ypred
mse=mean_squared_error(ytest,ypred)
mse
mae=mean_absolute_error(ytest,ypred)
mae
r2=r2_score(ytest,ypred)
r2
dt.predict([[5,6]])
```

## Output:
![image](https://github.com/user-attachments/assets/4ca7d5b3-0389-4111-bbaf-e2c17916a30d)

![image](https://github.com/user-attachments/assets/78f1b125-4ea0-4247-a719-e89bf2f21681)

![image](https://github.com/user-attachments/assets/6cc59a28-3323-4954-8d07-1d19865992c1)

![image](https://github.com/user-attachments/assets/0e4fd705-0f7f-4770-a46b-ebb31cf48a3d)

![image](https://github.com/user-attachments/assets/180375f3-7001-45f6-9d5f-645ccacad4b1)

![image](https://github.com/user-attachments/assets/4b28235f-b194-4c80-ba52-6b94d245895a)

![image](https://github.com/user-attachments/assets/1252793f-ab34-48da-9d65-0e530d3eb1e8)

![image](https://github.com/user-attachments/assets/3e101a36-039b-4e2b-97f3-82707fba3a05)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
