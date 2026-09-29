import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    confusion_matrix,
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# Load Datasets
df=pd.read_csv('student_exam_performance-selected-columns.csv')

print(df.head())
print("-"* 60)
print(df.tail())
print("-"* 60)
print(df.describe())
print("-"* 60)
print(df.info)
print("-"* 60)
print(df.shape)
print("-"* 60)
print(df.isnull().sum())
print("-"* 60)
print(df.dtypes)



sns.countplot(x='gender', data=df)
plt.show()
sns.scatterplot(x='gender', y='previous_gpa', data=df)
plt.show()

df = pd.get_dummies(df, columns=["gender"], dtype=int)
print(df.head())

df['parent_education'] = df['parent_education'].fillna(
    df['parent_education'].mode()[0]
)
df['previous_gpa']=df['previous_gpa'].fillna(df['previous_gpa'].mean())
print(df.isnull().sum())

x=df.drop(['student_id','education_level','school_type','family_income','parent_education','urban_rural','previous_gpa'],axis=1)
y=df['previous_gpa']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


lr = LinearRegression()

lr.fit(x_train, y_train)

predictions = lr.predict(x_test)

print("Predictions:")
print(predictions)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("-" * 60)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

print("-" * 60)

