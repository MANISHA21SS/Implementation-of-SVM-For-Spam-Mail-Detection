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

x=data["v1"].values
y=data["v2"].values

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

![Screenshot 2025-05-25 202422](https://github.com/user-attachments/assets/f44ea387-2427-42fb-ac57-f64c45c73536)


![Screenshot 2025-05-25 202437](https://github.com/user-attachments/assets/de7ef98e-d54c-4850-b3a3-fe448870d969)


![Screenshot 2025-05-25 202444](https://github.com/user-attachments/assets/cfa1a593-c465-474e-80a2-2a407e4b132e)


![Screenshot 2025-05-25 202453](https://github.com/user-attachments/assets/52e23729-8651-4a46-b740-c8deb277e77c)


![Screenshot 2025-05-25 202503](https://github.com/user-attachments/assets/0e875f4a-24a3-44fb-b7f1-5484afc23aa1)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
