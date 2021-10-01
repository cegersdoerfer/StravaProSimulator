import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import math
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import joblib


activities_df = pd.read_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/test_all_male_intervals_8-25")


scale_x = StandardScaler()
scale_y = StandardScaler()

X = activities_df[['e_gain', 'e_loss', 'turns', 'distance', 'intDistance', 'intE_gain', 'intE_loss', 'intTurns']]

y = activities_df[['intTime']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state = 1)


scaled_x_train = np.array(scale_x.fit_transform(X_train))
scaled_x_test = np.array(scale_x.transform(X_test))
scaled_y_train = np.array(scale_y.fit_transform(y_train)).flatten()
scaled_y_test = np.array(scale_y.transform(y_test)).flatten()

scaler_x_filename = "scale_x.save"
scaler_y_filename = "scale_y.save"
joblib.dump(scale_x, "/Users/chris_egersdoerfer/Desktop/SVR_Model/" + scaler_x_filename) 
joblib.dump(scale_y, "/Users/chris_egersdoerfer/Desktop/SVR_Model/" + scaler_y_filename)


regr = SVR(kernel = 'poly', degree = 1, C = 6, epsilon = .1)
print(cross_val_score(regr, scaled_x_train, scaled_y_train, cv=10, scoring = "r2").mean())
regr = SVR(kernel = 'rbf', degree = 1, C = 6, epsilon = .3)
print(cross_val_score(regr, scaled_x_train, scaled_y_train, cv=10, scoring = "r2").mean())
regr = SVR(kernel = 'poly', degree = 1, C = 6, epsilon = .5)
print(cross_val_score(regr, scaled_x_train, scaled_y_train, cv=10, scoring = "r2").mean())


scaled_y_train = scaled_y_train.reshape([len(scaled_y_train), 1])
scaled_y_test = scaled_y_test.reshape([len(scaled_y_test), 1])


regr = SVR(kernel = 'poly', degree = 1, C = 6, epsilon = .1)



regr.fit(scaled_x_train, np.transpose(scaled_y_train)[0])
result = regr.predict(scaled_x_test)

joblib.dump(regr, "/Users/chris_egersdoerfer/Desktop/SVR_Model/SVRTest.joblib")

#importance = regr.coef_
# summarize feature importance
#for i,v in enumerate(importance):
	#print('Feature: ' + str(i) + 'Score: ' + str(v))


scaled_y_train = scale_y.inverse_transform(scaled_y_train)
scaled_x_train = scale_x.inverse_transform(scaled_x_train)

result = scale_y.inverse_transform(result)
scaled_y_test = scale_y.inverse_transform(scaled_y_test)
scaled_x_test = scale_x.inverse_transform(scaled_x_test)
meanPrediction = sum(result)/len(result)
meanTest = sum(scaled_y_test)/len(scaled_y_test)
print("predicted mean: " + str(meanPrediction))
print("actual mean: " + str(meanTest))

variancePredict1 = 0
varianceReal1 = 0
for i in range(len(result)):
	variancePredict1 += (result[i] - meanPrediction)**2
	varianceReal1 += (scaled_y_test[i] - meanTest)**2
variancePredict1 = math.sqrt(variancePredict1/(len(result)-1))
varianceReal1 = math.sqrt(varianceReal1/(len(result)-1))

totalError = 0
totalPercentError = 0
for i in range(len(result)):
	currentError = abs(scaled_y_test[i] - result[i])
	currentPercentError = result[i] / scaled_y_test[i]
	if currentPercentError < 1:
		currentPercentError = 2 - currentPercentError
	totalError += currentError
	totalPercentError += currentPercentError

meanError = totalError / len(result)
meanPercentError = ((totalPercentError / len(result)) - 1) * 100
print("variance predictions: " + str(variancePredict1))
print("variance real: " + str(varianceReal1))
print("mean error: " + str(meanError))
print("mean percent error: " + str(meanPercentError[0].round(2)) + "%")
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(range(len(result)), scaled_y_test, color = 'r')
ax.plot(range(len(result)), result, color = 'b')
ax2 = fig.add_subplot(2,2,2)
ax2.plot(range(len(result)), np.transpose(scaled_x_test)[0], color = 'b')
ax3 = fig.add_subplot(2,2,3)
ax3.scatter(np.transpose(scaled_x_test)[4], scaled_y_test, color = 'g')
ax3.scatter(np.transpose(scaled_x_test)[4], result, color = 'r')
#ax3.scatter(np.transpose(scaled_x_train)[2], scaled_y_train, color = 'g')
ax3 = fig.add_subplot(2,2,4)
ax3.scatter(np.transpose(scaled_x_test)[1], scaled_y_test, color = 'g')
ax3.scatter(np.transpose(scaled_x_test)[1], result, color = 'r')
#ax3.scatter(np.transpose(scaled_x_train)[3], scaled_y_train, color = 'g')
plt.show()

print('Mean Squared Error:', math.sqrt(metrics.mean_squared_error(scaled_y_test, result)))











