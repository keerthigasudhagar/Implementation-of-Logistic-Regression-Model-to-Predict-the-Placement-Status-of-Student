# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.**
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results
## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Keerthika.S
RegisterNumber: 212223040093

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## DATA HEAD
![424285256-f78e5f92-9dd3-4f39-8dae-a01edbfb1e08](https://github.com/user-attachments/assets/bc7906b2-1aa4-4937-84ac-6e5e926022d8)

## DATA1 HEAD

![424285383-3920fd1a-44d4-4c79-8296-8212ae774093](https://github.com/user-attachments/assets/80f999d2-43c9-4f5c-89a0-01e32cdf01f3)

## ISNULL().SUM()
![424285571-b5b504eb-15d8-4274-b718-92f1d6b275b6](https://github.com/user-attachments/assets/d4f93ef6-63a1-45d2-970a-349898056bc6)

## DATA DUPLICATE

![424285703-cb6f4829-ee11-4252-8fa8-127cbd87697e](https://github.com/user-attachments/assets/7ddaa533-623d-4c42-bbca-7946d5384906)

## PRINT DATA
![424286046-f4d8def8-3789-4f61-9ef8-6fb060718f6b](https://github.com/user-attachments/assets/300b95d2-b03c-461b-8a76-b346ca0af8d8)

## STATUS
![424286306-f22c4c5c-1e60-437b-a4c7-02e198c73110](https://github.com/user-attachments/assets/7694994a-6a12-452f-bbde-ce6c3d98e218)

## Y_PRED
![424286707-9d396877-4d02-4ed4-92ea-dbbe7534ee1a](https://github.com/user-attachments/assets/bd10f04a-60a1-4b95-a08a-6f7598587a7d)

## ACCURACY
![424286760-bdfa007e-3ee3-4d82-8b4d-378b9ae3d0d5](https://github.com/user-attachments/assets/02eccf5e-63ff-4b6b-9be9-ba38bc6d0f80)

## CONFUSION MATRIX
![424286925-6da6ad6a-66fa-43f7-a6d4-56363113f979](https://github.com/user-attachments/assets/c083d833-979b-4258-93f9-e2279971a615)

## CLASSIFICATION

![424287100-b1398038-74d4-415b-8d4d-700ceb1bb86f](https://github.com/user-attachments/assets/c20bc5af-d05d-4f58-80e0-bc8217d76ddb)

## LR PREDICT
![424287622-610db60b-96c3-4c75-8694-7e7394d3e188](https://github.com/user-attachments/assets/f177580c-883c-4cde-b24f-e77c53855660)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
