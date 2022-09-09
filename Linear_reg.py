

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #creation of the object
regressor.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#To predict a specific salary
regressor.predict([[15]])

# Visualising the Training set results
plt.scatter(X_test, y_test, color = 'red')

#to plot the observation points

#pour tracer les pts d'observation

plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


plt.show() 

# the red crosses are real values 
# the blue line corresponds to our linear regression model
# to predict we project the red points on the line, we make another projection of this point on the salary abscissa to have the predicted salary

plt.show()
#les crois en rouge sont des valeurs reels 
#la droite bleue corespand a notre modele de regression lineaire
#pour predire on fait la projection des points rouges sur la droite, on fait une autre projection de ce point sur l abscisse salary pour avoir le salaire predit

