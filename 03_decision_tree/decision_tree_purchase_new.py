import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report

# Load data
df = pd.read_csv('Purchase_new.csv')

print(df.head())
print('-' * 60)

print(df.describe())
print('-' * 60)

print(df.columns)
print('-' * 60)

print(df.shape)
print('-' * 60)

print(df.info())
print('-' * 60)

print(df.isnull().sum())
print('-' * 60)


# Visualization
sns.histplot(df['Discount'])
plt.show()

sns.barplot(
    x=df['Discount'],
    y=df['Purchase']
)
plt.show()


# Encoding
le = LabelEncoder()

df['Discount'] = le.fit_transform(df['Discount'])
df['Holiday'] = le.fit_transform(df['Holiday'])
df['Free Delivery'] = le.fit_transform(df['Free Delivery'])
df['Purchase'] = le.fit_transform(df['Purchase'])

print(df.dtypes)
print(df['Discount'].value_counts())
print(df['Discount'].nunique())
print(df['Discount'].unique())

# Features and Target
X = df.drop('Discount', axis=1)
y = df['Discount']

print(df['Discount'].value_counts())
print(df['Discount'].nunique())
print(df['Discount'].unique())

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)


# Prediction
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)


# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(cm)

cls=classification_report(y_test, predictions)
print("Classification Report:",cls)

# Heatmap
sns.heatmap(cm, annot=True, fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()