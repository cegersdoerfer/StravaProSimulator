from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator
import random
import proSimulator



def convert_real_tracks(all_activities, simulator, dist_range):
	measure_df = pd.DataFrame(columns = ["e_gain", "e_loss", "distance", 
										 "turns", 
										 "intE_gain", "intE_loss","intDistance", 
										 "intTurns"])
	for name, activity in all_activities.items():
		activity = simulator.findAllMeasures(activity, Time = False, Speed = False, Slope = False, columnLabels = {'dist': "distance", 
																												   'e_gain': "e_gain", 
																												   'e_loss': "e_loss"})
		
		routes = []
		points = []
		for r in range(len(activity)):
				row = activity.iloc[r]
				points.append([row.Longitude, row.Latitude])
		turns = simulator.findTurns(points)

		lastRow = activity.iloc[-1,:]

		max_dist = dist_range[1]
		min_dist = dist_range[0]
		normalized_dist = (lastRow['distance']-min_dist)/(max_dist - min_dist)
		normalized_dist = round(normalized_dist * 20)
		if normalized_dist == 0:
			normalized_dist += 1

		activityIntervals = simulator.findIntervalsByNum(activity, intervalCount = normalized_dist)

		routes.append(activityIntervals)
		activityLastRow = lastRow.copy()
		activityLastRow.loc["turns"] = len(turns[0])
		activityLastRow = activityLastRow[["e_gain", "e_loss", "distance", "turns"]]

		for interval in activityIntervals:
			interval = simulator.findAllMeasures(activityIntervals[interval], Time = False, 
												 Speed = False, columnLabels = {'dist': "intDistance", 
																				'e_gain': "intE_gain", 
																				'e_loss': "intE_loss", 
																				'slope':"intSlope"})
			points = []
			for r in range(len(interval)):
				row = interval.iloc[r]
				points.append([row.Longitude, row.Latitude])
			turns = simulator.findTurns(points)

			lastRow = interval.iloc[-1,:]
			lastRow = lastRow[["intE_gain", "intE_loss", "intDistance"]]
			lastRow.loc["e_gain"] = activityLastRow["e_gain"]
			lastRow.loc["e_loss"] = activityLastRow["e_loss"]
			lastRow.loc["distance"] = activityLastRow["distance"]
			lastRow.loc["turns"] = activityLastRow["turns"]
			lastRow.loc["intTurns"] = len(turns[0])


			measure_df = measure_df.append(lastRow, ignore_index = True)
	
	print(measure_df)

	return (measure_df, routes)



def predict_track(test_path = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/testRoutes"):

	pds = ProDataSimulator(path = test_path, simulate = True)


	all_activities = pds.getAllActivities()
	activities_df_and_route = convert_real_tracks(all_activities, pds, [10575, 244273])
	activities_df = activities_df_and_route[0]
	routes = activities_df_and_route[1]

	scale_x = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/scale_x.save")
	scale_y = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/scale_y.save")

	X = activities_df[['e_gain', 'e_loss', 'turns', 'distance', 'intDistance', 'intE_gain', 'intE_loss', 'intTurns']]
	scaled_x = scale_x.transform(X)

	regr = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/SVRTest.joblib")

	result = regr.predict(scaled_x)
	all_results = []
	res_list = scale_y.inverse_transform(result)
	all_results.append(res_list)

	totalTime = 0
	for res in res_list:
		totalTime += res

	dist_row = activities_df.iloc[-1]
	dist = dist_row.loc['distance']

	speed = dist/totalTime

	return totalTime, speed

		

if __name__ == "__main__":
	test_route_dir = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/testRoutes"
	results = predict_track(test_route_dir)
	print(results[0]/3600)
	print(results[1])



		



