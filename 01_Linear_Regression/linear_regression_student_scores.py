import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_csv('rounded_hours_student_scores.csv')

print(train.head())
print('-'*60)
print(train.tail())
print('-'*60)
print(train.describe())
print('-'*60)
print(train.info())
print('-'*60)
print(train.columns)
print('-'*60)

sns.scatterplot(data=train, x='Hours', y='Scores')

plt.title('Hours vs Scores')
plt.xlabel('Study Hours')
plt.ylabel('Scores')
plt.show()


sns.heatmap(train.corr(), annot=True)
plt.show()


X=train.drop('Scores', axis=1)
y=train['Scores']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)


lm=LinearRegression()
lm.fit(X_train, y_train)


print("Intercept: ", lm.intercept_)


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(X_test)
print("Sample Predcition : ",predictions[:10])
print('-'*60)

plt.scatter(y_test, predictions)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.show()