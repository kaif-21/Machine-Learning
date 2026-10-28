import pandas as pd
import numpy as np
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

# load datasets
df = pd.read_csv("cars_dataset.csv")



# data visualization
print(df.head())
print("-"*60)
print(df.tail())
print("-"*60)
print(df.describe()) # mean | meadian | STD Deviation ....
print("-"*60)
print((df.info())) # columns | not null ....
print("-"*60)
print(df.columns)
print("-"*60)
print(df.dtypes)


# Apply EDA

sns.histplot(df['buying'])
plt.show()

sns.histplot(df['maint'])
plt.show()

sns.scatterplot(x=df['buying'], y=df['car'])
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

print(df.dtypes)

# define x and y

x=df.drop(['car'], axis=1)
y=df['car']

# train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# apply decision tree algorithm

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

accuracy=accuracy_score(y_test, y_pred)
print(accuracy)

cls=classification_report(y_test, y_pred)
print(cls)

sns.heatmap(cm, annot=True, fmt="d")
plt.show()








