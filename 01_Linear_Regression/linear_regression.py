# Dataset --> USA House price prediction

# STEP 1 --> Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries --> Sckit learn --> Model | function  | metrices | classes
from sklearn.model_selection import train_test_split # used for data splitting into training and testing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Step 2 --> Load the Dataset
df = pd.read_csv("USA_Housing.csv")

# Step 3 --> Data Exploration
print(df.head())
print("-"*60)
print(df.tail())
print("-"*60)
print(df.describe()) # mean | meadian | STD Deviation ....
print("-"*60)
print((df.info())) # columns | not null ....
print("-"*60)
print(df.columns)

# Step 4 --> Data Visualization

# Pair plot

sns.pairplot(df)
plt.show()

sns.histplot(df['Price'], kde=True)
plt.show()

sns.heatmap(
    df.drop("Address", axis=1).corr(),
    annot=True,
    cmap="coolwarm" # coloring
)
plt.show()

# Step 5 -- .Preparing data for training
X = df.drop(['Price', 'Address'], axis=1)  # Independent variable | features | input
y = df['Price'] # dependent variable  | label | target | output

# Step 6 --> Data divide into training(more data usually 70%) and testing(30%)
# mid term 1 marks --> 25/30 mid term 2 marks --> 28/30 | classes | preparation | practicles --> main exam 9.7(output )
# Error --> actual - predicted
# on which data  we will calculate the error of the model --> testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) # random_state --> reproducibility

# Step 7 --> Model initialization
lm = LinearRegression()

# step 8 --> Model training
lm.fit(X_train, y_train) # only training data

# Linear regression formula --> Y = mX + C  --> m = Slop  C = intercept

print("Intercept: ", lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Make predictions
predictions = lm.predict(X_test)
print("Sample Predcition : ",predictions[:10])

# Actual vs predicted value difference

plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

# HW --> MAE  | MSE | RMSE

mae = mean_absolute_error(y_test, y_test)

mse = mean_squared_error(y_test, y_test)

rmse = mean_squared_error(y_test, y_test) ** 0.5

