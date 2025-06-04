# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program

## Program:
/*
Program to implement the SVM For Spam Mail Detection..

Developed by: Manisha selvakumari.S.S.

RegisterNumber: 212223220055

*/
```
from google.colab import files
uploaded = files.upload()

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

y=data["v2"].values
x=data["v1"].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![image](https://github.com/user-attachments/assets/4cf2253f-f0f0-479f-93ef-741cf8a601b4)

![image](https://github.com/user-attachments/assets/dcdcecfd-99dc-4237-9b41-be7f09c33fd9)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
