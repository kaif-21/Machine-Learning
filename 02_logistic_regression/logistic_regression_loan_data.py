import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    confusion_matrix,
    accuracy_score,
    classification_report,
    recall_score,
)
from streamlit import columns

df=pd.read_csv('Loan Eligibility Prediction-selected-columns.csv')

print(df.head())
print('-'*60)
print(df.tail())
print('-'*60)
print(df.describe())
print('-'*60)
print(df.columns)
print('-'*60)
print(df.shape)
print('-'*60)
print(df.info())
print('-'*60)
print(df.isnull().sum())
print('-'*60)


# sns.histplot(df['person_age'])
# plt.show()

# sns.countplot(x='loan_amnt',data=df)
# plt.show()

# Column names ke extra spaces remove karo
df.columns = df.columns.str.strip()

le = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for column in categorical_columns:
    df[column] = le.fit_transform(df[column].astype(str))
print(df.dtypes)


X = df.drop(["Self_Employed ","Education"], axis=1)
y = df["Self_Employed"]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgr =LogisticRegression()
lgr.fit(x_train, y_train)

