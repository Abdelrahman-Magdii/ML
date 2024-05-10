import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv('./archive/wages_cleaned.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into features (x) and target variable (y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)



# Visualising the Training set results
plt.scatter(Y_train, regressor.predict(X_train), color='blue')
plt.plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='red', linestyle='--')
plt.title('Actual vs Predicted Salary (Training set)')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.show()


# Visualising the Testing set results
plt.scatter(Y_test, regressor.predict(X_test), color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Salary (Testing set)')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.show()



plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - Y_train, c="blue", label="Training Data")
plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - Y_test, c="orange", label="Testing Data")
plt.legend()
plt.hlines(y=0, xmin=y.min(), xmax=y.max())
plt.title("Residual Plot")
plt.show()
