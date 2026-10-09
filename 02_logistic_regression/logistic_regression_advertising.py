import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,  # gives Precision, Recall, F1-Score, Support
    confusion_matrix,  # Shows correct and incorrect predictions
    accuracy_score,  # gives overall accuracy of the model

)

train = pd.read_csv("advertising.csv")
print("-"*60)
print(train.head())
print("-"*60)
print(train.columns)
print('-'*60)
print(train.describe())
print("-"*60)
print(train.info())
print("-"*60)
print(train.tail())
print("-"*60)
print(train.shape)
print("-"*60)
print('\nMissing Values:')
print(train.isnull().sum())
print("-"*60)


print("-"*60)




train.drop(["Ad Topic Line","City","Male","Country","Timestamp"], axis=1, inplace=True)
print(train.head())

X=train.drop("Clicked on Ad",axis=1)
y=train["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)



predictions = logmodel.predict(X_test)

print('\n Accuracy:' )
print('\nAccuracy:', accuracy_score(y_test, predictions))


print('\nConfusion Matrix')
cm = confusion_matrix(y_test,predictions)
print("\nConfusion Matrix: \n", cm)
print('-'*60)

print('\nClassification Report')
classification_report(y_test,predictions)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted ad')
plt.ylabel('Actual ad')
plt.show()





