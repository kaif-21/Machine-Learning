import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split  # to split data into training or testing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,                            # gives Precision, Recall, F1-Score, Support
    confusion_matrix,                                 # Shows correct and incorrect predictions
    accuracy_score,                                   # gives overall accuracy of the model

)

train = pd.read_csv("titanic_train (1) (3).csv")
print("-"*60)
print(train.head())
print("-"*60)
print(train.tail())
print("-"*60)
print(train.describe())
print("-"*60)
print(train.info())
print("-"*60)
print(train.shape)
print("-"*60)
print(train.columns)
print("-"*60)
print("\nMisssing value before cleaning")
print(train.isnull().sum())
print("-"*60)

sns.heatmap(
    train.isnull(),
    yticklabels=False,  # remove labels from The y-axis
    cbar=False,         #removes color from the side of heatmap
    cmap="viridis"
)
plt.show()
print("-"*60)

def impute_age(raw):
    if pd.isnull(raw["Age"]):
        if raw['Pclass']==1:
            return 37
        elif raw['Pclass']==2:
            return 28
        else:
            return 24
    else:
        return raw["Age"]


# apply the function into the dataset
train['age'] = train.apply(impute_age,axis=1)
print("-"*60)

# Remove unwanted columns
train.drop('Cabin',axis=1,inplace=True)

# Remove null value
train.dropna(inplace=True)
print("-"*60)

# converting Gender into numeric data
sex=pd.get_dummies(train['Sex'],drop_first=True)   # 0--> Female  1--> Male
sex.astype(int)
print("-"*60)

# Converting Embarks into numeric data
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.astype(int)
print("-"*60)

# Attenching the dummy data into original data
train=pd.concat([train, embark], axis=1)
print("-"*60)

# Removing the old columns
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
print("-"*60)

print('\nClean data')
print(train.head())
print("-"*60)

sns.heatmap(
    train.isnull(),
    yticklabels=False, # remove labels from The y-axis
    cbar=False, # removes color from the side of heatmap
    cmap='viridis'
)
plt.show()
print("-"*60)

# Creating Feature and Labels
X=train.drop('Survived',axis=1) # --> input columns
Y=train['Survived']  # ---> target column

# Training and Testing Splitting
print('\nTraining and Testing Splitting')
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=40)

# Model apply
print('\nModel apply')
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)

# Prediction
print('\nPrediction')
predictions = logmodel.predict(X_test)
print("-"*60)

# Accuracy
print('\nAccuracy')
print('\n Accuracy:',accuracy_score(Y_test,predictions))
print("-"*60)

# Confusion Matrics
cm = confusion_matrix(Y_test,predictions)
print("\nConfusion Matrix: ", cm)
print("-"*60)

# Classification Report
print('\nClassification Report')
classification_report(Y_test,predictions)
print("-"*60)

# Confusion matrics with graph
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("-"*60)

example_p_details = pd.DataFrame(
    [[1, 27, 5, 1, 10, 1, 1, 0]],
    columns=X.columns
)
print('\nExample Passengar')
print(example_p_details)
print("-"*60)

# Chance of the survivor

print("Prediction (0 = Margyi), (1 = Bachgi) : ", logmodel.predict(example_p_details))
print("-"*60)
print("Prediction Probability: ", logmodel.predict_proba(example_p_details))












