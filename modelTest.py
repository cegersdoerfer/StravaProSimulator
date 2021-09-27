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


activities_df = pd.read_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/test_all_male_intervals")

"""
#X = activities_df[['e_gain', 'e_loss', 'distance', 'turns']]
X = activities_df[['e_gain', 'e_loss', 'distance', 'turns', 'intDistance', 'intE_gain', 'intE_loss']]
y = activities_df['intTime'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators = 1)

regr.fit(X_train, y_train)
result = regr.predict(X_test)
print(result)
print(y_test)
meanPrediction = sum(result)/len(result)
meanTest = sum(y_test)/len(y_test)
print("predicted mean: " + str(meanPrediction))
print("actual mean: " + str(meanTest))
totalError = 0
for i in range(len(result)):
	currentError = abs(y_test[i] - result[i])
	totalError += currentError

meanError = totalError / len(result)
print("mean error: " + str(meanError))
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(range(len(result)), y_test, color = 'r')
ax.plot(range(len(result)), result, color = 'b')
ax2 = fig.add_subplot(2,2,2)
ax2.plot(range(len(result)), X_test, color = 'b')
ax3 = fig.add_subplot(2,2,3)
ax3.scatter(X_test.iloc[:, 4], y_test, color = 'g')
ax3.scatter(X_test.iloc[:, 4], result, color = 'r')
#ax3.scatter(np.transpose(scaled_x_train)[2], scaled_y_train, color = 'g')
ax3 = fig.add_subplot(2,2,4)
ax3.scatter(X_test.iloc[:, 5], y_test, color = 'g')
ax3.scatter(X_test.iloc[:, 5], result, color = 'r')
#ax3.scatter(np.transpose(scaled_x_train)[3], scaled_y_train, color = 'g')
plt.show()


print('Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, result)))


"""

scale_x= StandardScaler()
scale_y= StandardScaler()

X = activities_df[['e_gain', 'e_loss', 'distance', 'turns', 'intDistance', 'intE_gain', 'intE_loss']]
#X = activities_df[['intDistance', 'intE_gain', 'intE_loss']]

y = activities_df[['intTime']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state = 3)


scaled_x_train = np.array(scale_x.fit_transform(X_train))
scaled_x_test = np.array(scale_x.transform(X_test))
scaled_y_train = np.array(scale_y.fit_transform(y_train)).flatten()
scaled_y_test = np.array(scale_y.transform(y_test)).flatten()


regr = SVR(kernel = 'poly', degree=1, C = .4, epsilon = .5)
print(cross_val_score(regr, scaled_x_train, scaled_y_train, cv=10, scoring = "r2").mean())
regr = SVR(kernel = 'linear', degree=1, C = .4, epsilon = .5)
print(cross_val_score(regr, scaled_x_train, scaled_y_train, cv=10, scoring = "r2").mean())
regr = SVR(kernel = 'rbf', degree=1, C = .4, epsilon = .5)
print(cross_val_score(regr, scaled_x_train, scaled_y_train, cv=10, scoring = "r2").mean())


scaled_y_train = scaled_y_train.reshape([len(scaled_y_train), 1])
scaled_y_test = scaled_y_test.reshape([len(scaled_y_test), 1])


regr = SVR(kernel = 'rbf', degree = 1, C = .4, epsilon = .5)



regr.fit(scaled_x_train, np.transpose(scaled_y_train)[0])
result = regr.predict(scaled_x_test)

joblib.dump(regr, "/Users/chris_egersdoerfer/Desktop/SVR_Model/SVR.joblib")

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
ax3.scatter(np.transpose(scaled_x_test)[0], scaled_y_test, color = 'g')
ax3.scatter(np.transpose(scaled_x_test)[0], result, color = 'r')
#ax3.scatter(np.transpose(scaled_x_train)[2], scaled_y_train, color = 'g')
ax3 = fig.add_subplot(2,2,4)
ax3.scatter(np.transpose(scaled_x_test)[1], scaled_y_test, color = 'g')
ax3.scatter(np.transpose(scaled_x_test)[1], result, color = 'r')
#ax3.scatter(np.transpose(scaled_x_train)[3], scaled_y_train, color = 'g')
plt.show()

print('Mean Squared Error:', math.sqrt(metrics.mean_squared_error(scaled_y_test, result)))






"""

X = activities_df[['e_gain', 'distance', 'turns']]
y = activities_df[['time']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 3)


kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X_train)
print(kmeans.cluster_centers_)
X_train['cluster'] = kmeans.labels_
X_train['y'] = y_train['time'].tolist()

cluster_predictions = kmeans.predict(X_test)

X_test['cluster'] = cluster_predictions
X_test['y'] = y_test['time'].tolist()

X_train_cluster1 = X_train[X_train['cluster'] == 0]
y_train_cluster1 = pd.DataFrame(X_train_cluster1['y'].tolist())
y_train_cluster1.columns = ['time']
X_train_cluster1 = X_train_cluster1.drop(columns=['cluster', 'y'], axis = 1)

X_train_cluster2 = X_train[X_train['cluster'] == 1]
y_train_cluster2 = pd.DataFrame(X_train_cluster2['y'].tolist())
y_train_cluster2.columns = ['time']
X_train_cluster2 = X_train_cluster2.drop(columns=['cluster', 'y'], axis = 1)

X_test_cluster1 = X_test[X_test['cluster'] == 0]
y_test_cluster1 = pd.DataFrame(X_test_cluster1['y'].tolist())
y_test_cluster1.columns = ['time']
X_test_cluster1 = X_test_cluster1.drop(columns=['y', 'cluster'], axis = 1)

X_test_cluster2 = X_test[X_test['cluster'] == 1]
y_test_cluster2 = pd.DataFrame(X_test_cluster2['y'].tolist())
y_test_cluster2.columns = ['time']
X_test_cluster2 = X_test_cluster2.drop(columns=['y', 'cluster'], axis = 1)


scale_X1= StandardScaler()
scale_X2= StandardScaler()
scale_y1= StandardScaler()
scale_y2= StandardScaler()
scaled_x_train_cluster1 = np.array(scale_X1.fit_transform(X_train_cluster1))
scaled_x_train_cluster2 = np.array(scale_X2.fit_transform(X_train_cluster2))
scaled_x_test_cluster1 = np.array(scale_X1.transform(X_test_cluster1))
scaled_x_test_cluster2 = np.array(scale_X2.transform(X_test_cluster2))
scaled_y_train_cluster1 = np.array(scale_y1.fit_transform(y_train_cluster1))
scaled_y_train_cluster2 = np.array(scale_y2.fit_transform(y_train_cluster2))
scaled_y_test_cluster1 = np.array(scale_y1.transform(y_test_cluster1))
scaled_y_test_cluster2 = np.array(scale_y2.transform(y_test_cluster2))
print(scaled_y_test_cluster1)
print(scaled_y_test_cluster2)




#print(scaled_x_train)
#print(scaled_x_test)
#print(scaled_y_train)
#print(scaled_y_test)
#print(scale_test.inverse_transform(scaled_y_test))

regr_cluster1 = SVR(kernel = 'poly', degree=1, C = .4, epsilon=.5)
regr_cluster2 = SVR(kernel = 'poly', C = .5)


scores_cluster1 = cross_val_score(regr_cluster1, scaled_x_train_cluster1, np.transpose(scaled_y_train_cluster1)[0], cv=10, scoring='explained_variance')
print(scores_cluster1)
scores_cluster2 = cross_val_score(regr_cluster2, scaled_x_train_cluster2, np.transpose(scaled_y_train_cluster2)[0], cv=10, scoring='explained_variance')
print(scores_cluster2)

regr_cluster1.fit(scaled_x_train_cluster1, np.transpose(scaled_y_train_cluster1)[0])
regr_cluster2.fit(scaled_x_train_cluster2, np.transpose(scaled_y_train_cluster2)[0])
result1 = regr_cluster1.predict(scaled_x_test_cluster1)
result2 = regr_cluster2.predict(scaled_x_test_cluster2)
print(result1)
print(result2)

result1 = scale_y1.inverse_transform(result1)
result2 = scale_y2.inverse_transform(result2)
scaled_y_test1 = scale_y1.inverse_transform(scaled_y_test_cluster1)
scaled_y_test2 = scale_y2.inverse_transform(scaled_y_test_cluster2)

meanPrediction_cluster1 = sum(result1)/len(result1)
meanPrediction_cluster2 = sum(result2)/len(result2)

meanTest_cluster1 = sum(scaled_y_test1)/len(scaled_y_test1)
meanTest_cluster2 = sum(scaled_y_test2)/len(scaled_y_test2)

print("predicted mean cluster 1: " + str(meanPrediction_cluster1))
print("predicted mean cluster 2: " + str(meanPrediction_cluster2))
print("actual mean cluster 1: " + str(meanTest_cluster1))
print("actual mean cluster 2: " + str(meanTest_cluster2))

variancePredict1 = 0
varianceReal1 = 0
for i in range(len(result1)):
	variancePredict1 += (result1[i] - meanPrediction_cluster1)**2
	varianceReal1 += (scaled_y_test1[i] - meanTest_cluster1)**2
variancePredict1 = math.sqrt(variancePredict1/(len(result1)-1))
varianceReal1 = math.sqrt(varianceReal1/(len(result1)-1))

variancePredict2 = 0
varianceReal2 = 0
for i in range(len(result2)):
	variancePredict2 += (result2[i] - meanPrediction_cluster2)**2
	varianceReal2 += (scaled_y_test2[i] - meanTest_cluster2)**2
variancePredict2 = math.sqrt(variancePredict2/(len(result2)-1))
varianceReal2 = math.sqrt(varianceReal2/(len(result2)-1))

totalError1 = 0
totalPercentError1 = 0
for i in range(len(result1)):
	currentError = abs(scaled_y_test1[i] - result1[i])
	currentPercentError = result1[i]/scaled_y_test1[i]
	if currentPercentError < 1:
		currentPercentError = 2 - currentPercentError
	totalError1 += currentError
	totalPercentError1 += currentPercentError


totalError2 = 0
totalPercentError2 = 0
for i in range(len(result2)):
	currentError = abs(scaled_y_test2[i] - result2[i])
	currentPercentError = result2[i] / scaled_y_test2[i]
	if currentPercentError < 1:
		currentPercentError = 2 - currentPercentError
	totalError2 += currentError
	totalPercentError2 += currentPercentError

meanError_cluster1 = totalError1 / len(result1)
meanError_cluster2 = totalError2 / len(result2)
meanPercentError_cluster1 = ((totalPercentError1 / len(result1))-1)*100
meanPercentError_cluster2 = ((totalPercentError2 / len(result2))-1)*100

print("mean error cluster 1: " + str(meanError_cluster1))
print("mean error cluster 2: " + str(meanError_cluster2))
print("mean percent error cluster 1: " + str(meanPercentError_cluster1[0].round(2)) + "%")
print("mean percent error cluster 2: " + str(meanPercentError_cluster2[0].round(2)) + "%")
print("variance predictions cluster1: " + str(variancePredict1))
print("variance predictions cluster2: " + str(variancePredict2))
print("variance real cluster1: " + str(varianceReal1))
print("variance real cluster2: " + str(varianceReal2))
print("avg mean Error: " + str((meanError_cluster2 + meanError_cluster1)/2))


fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(range(len(result1)), scaled_y_test1, color = 'r')
ax.plot(range(len(result1)), result1, color = 'b')

ax1 = fig.add_subplot(1,2,2)
ax1.plot(range(len(result2)), scaled_y_test2, color = 'r')
ax1.plot(range(len(result2)), result2, color = 'b')
plt.show()

"""










