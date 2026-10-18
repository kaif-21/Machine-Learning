import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    confusion_matrix,
    accuracy_score,
    classification_report
)

df=pd.read_csv("Heart Attack.csv")

print(df.head())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)

le = LabelEncoder()
df['class']=le.fit_transform(df['class'])
print(df.dtypes)

x=df.drop(['class'],axis=1)
y=df['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)

cls=classification_report(y_test,y_pred)
print(cls)

sns.heatmap(cm,annot=True,fmt='d')
plt.show()