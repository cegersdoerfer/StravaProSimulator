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
from WriteCSV import create_csv





pds = ProDataSimulator(path = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/testRoutes", simulate = True)


all_activities = pds.getAllActivities()
print(all_activities)
activities_df = create_csv(all_activities, pds, intervals = True, simulate = True, filePath = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/proData-csv/real_test.csv")

activities_df = pd.read_csv("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/proData-csv/real_test.csv")






scale_x = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/scale_x.save")
scale_y = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/scale_y.save")

X = activities_df[['e_gain', 'e_loss', 'turns', 'distance', 'intDistance', 'intE_gain', 'intE_loss', 'intTurns']]
#X = activities_df[['e_gain', 'e_loss', 'distance', 'turns']]

scaled_x = scale_x.transform(X)

regr = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/SVRTest.joblib")

result = regr.predict(scaled_x)
print(result)

res_list = scale_y.inverse_transform(result)
print(res_list)
totalTime = 0
for res in res_list:
	totalTime += res

print(totalTime)

print(totalTime/3600)
dist_row = activities_df.iloc[-1]
dist = dist_row.loc['distance']

speed = dist/totalTime
print(speed)

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.plot(range(len(result)), result, color = 'r')
#plt.show()


