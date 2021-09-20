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


activities_df = pd.read_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/test_all_male")

"""
X = activities_df[['e_gain', 'e_loss', 'distance', 'turns']]
y = activities_df['time'].tolist()
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
ax = fig.add_subplot(1,1,1)
ax.plot(range(len(result)), y_test, color = 'r')
ax.plot(range(len(result)), result, color = 'b')
plt.show()


print('Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, result)))
"""


"""

scale_train= StandardScaler()
scale_test= StandardScaler()

X = activities_df[['e_gain', 'distance', 'turns']]
y = activities_df[['time']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

scaled_x_train = np.array(scale_train.fit_transform(X_train))
scaled_x_test = np.array(scale_test.fit_transform(X_test))
scaled_y_train = np.array(scale_train.fit_transform(y_train))
scaled_y_test = np.array(scale_test.fit_transform(y_test))


#print(scaled_x_train)
#print(scaled_x_test)
#print(scaled_y_train)
#print(scaled_y_test)
#print(scale_test.inverse_transform(scaled_y_test))

regr = SVR(kernel = 'poly', C = .5)

regr.fit(scaled_x_train, np.transpose(scaled_y_train)[0])
result = regr.predict(scaled_x_test)


result = scale_test.inverse_transform(result)
scaled_y_test = scale_test.inverse_transform(scaled_y_test)
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
	print(currentPercentError)
	totalError += currentError
	totalPercentError += currentPercentError

meanError = totalError / len(result)
meanPercentError = ((totalPercentError / len(result)) - 1) * 100
print("variance predictions: " + str(variancePredict1))
print("variance real: " + str(varianceReal1))
print("mean error: " + str(meanError))
print("mean percent error: " + str(meanPercentError[0].round(2)) + "%")
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(len(result)), scaled_y_test, color = 'r')
ax.plot(range(len(result)), result, color = 'b')
plt.show()

print('Mean Squared Error:', math.sqrt(metrics.mean_squared_error(scaled_y_test, result)))


"""






X = activities_df[['e_gain', 'distance', 'turns', 'downHillTurns']]
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


scale_X_train1= StandardScaler()
scale_X_train2= StandardScaler()
scale_X_test1= StandardScaler()
scale_X_test2= StandardScaler()
scale_y_train1= StandardScaler()
scale_y_train2= StandardScaler()
scale_y_test1= StandardScaler()
scale_y_test2= StandardScaler()
scaled_x_train_cluster1 = np.array(scale_X_train1.fit_transform(X_train_cluster1))
scaled_x_train_cluster2 = np.array(scale_X_train2.fit_transform(X_train_cluster2))
scaled_x_test_cluster1 = np.array(scale_X_test1.fit_transform(X_test_cluster1))
scaled_x_test_cluster2 = np.array(scale_X_test2.fit_transform(X_test_cluster2))
scaled_y_train_cluster1 = np.array(scale_y_train1.fit_transform(y_train_cluster1))
scaled_y_train_cluster2 = np.array(scale_y_train2.fit_transform(y_train_cluster2))
scaled_y_test_cluster1 = np.array(scale_y_test1.fit_transform(y_test_cluster1))
scaled_y_test_cluster2 = np.array(scale_y_test2.fit_transform(y_test_cluster2))
print(scaled_y_test_cluster1)
print(scaled_y_test_cluster2)




#print(scaled_x_train)
#print(scaled_x_test)
#print(scaled_y_train)
#print(scaled_y_test)
#print(scale_test.inverse_transform(scaled_y_test))

regr_cluster1 = SVR(kernel = 'poly', C = .5)
regr_cluster2 = SVR(kernel = 'poly', C = .5)

regr_cluster1.fit(scaled_x_train_cluster1, np.transpose(scaled_y_train_cluster1)[0])
regr_cluster2.fit(scaled_x_train_cluster2, np.transpose(scaled_y_train_cluster2)[0])
result1 = regr_cluster1.predict(scaled_x_test_cluster1)
result2 = regr_cluster2.predict(scaled_x_test_cluster2)
print(result1)
print(result2)

result1 = scale_y_test1.inverse_transform(result1)
result2 = scale_y_test2.inverse_transform(result2)
scaled_y_test1 = scale_y_test1.inverse_transform(scaled_y_test_cluster1)
scaled_y_test2 = scale_y_test2.inverse_transform(scaled_y_test_cluster2)

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












