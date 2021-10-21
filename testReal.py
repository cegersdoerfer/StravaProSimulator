from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator
import random
import proSimulator
import numpy
import seaborn as sns



def convert_real_tracks(all_activities, simulator, dist_range, interval_split = False):
	measure_df = pd.DataFrame(columns = ["e_gain", "e_loss", "distance", 
										 "turns", 
										 "intE_gain", "intE_loss","intDistance", 
										 "intTurns"])
	interval_dict = {}
	for name, activity in all_activities.items():
		#print(name)
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

		if interval_split:
			activityIntervals = simulator.findIntervalsByNum(activity, intervalCount = 10)
		else:
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
		activityLastRow = activityLastRow[["e_gain", "e_loss", "distance", "turns", "Latitude", "Longitude"]]
		interval_dict.update({name: {}})
		for interval_name in activityIntervals:
			interval = simulator.findAllMeasures(activityIntervals[interval_name], Time = False, 
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
			lastRow.loc["Latitude"] = activityLastRow["Latitude"]
			lastRow.loc["Longitude"] = activityLastRow["Longitude"]


			measure_df = measure_df.append(lastRow, ignore_index = True)
			if interval_split is True:
				interval_dict[name][interval_name] = lastRow
			else:
				interval_dict[name][interval_name] = interval

	
	#print(measure_df)

	return (measure_df, interval_dict)



def predict_track(test_path = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/testRoutes"):

	pds = ProDataSimulator(path = test_path, simulate = True)


	all_activities = pds.getAllActivities()
	activities_df_and_route = convert_real_tracks(all_activities, pds, [10575, 244273])
	activities_df = activities_df_and_route[0]
	routes = activities_df_and_route[1]

	split_dict = {}
	for name in routes:
		print(name)
		broken_activities = convert_real_tracks(routes[name], pds, [10575, 244273], interval_split = True)
		split_dict[name] = broken_activities[1]

	scale_x = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/scale_x.save")
	scale_y = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/scale_y.save")

	X = activities_df[['e_gain', 'e_loss', 'turns', 'distance', 'intDistance', 'intE_gain', 'intE_loss', 'intTurns']]
	scaled_x = scale_x.transform(X)

	regr = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/SVRTest.joblib")

	result = regr.predict(scaled_x)
	res_list = scale_y.inverse_transform(result)

	int_scale_x = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/int_x_scalar.save")
	int_scale_y = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/int_y_scalar.save")

	X = activities_df[['e_gain', 'e_loss', 'turns', 'distance', 'intDistance', 'intE_gain', 'intE_loss', 'intTurns']]
	scaled_x = scale_x.transform(X)

	int_regr = joblib.load("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/SVR_Model/int_svr.joblib")
	# iterate through all tracks
	for name, intervals in split_dict.items():
		interval_results = []
		interval_distances = []
		predict_track_df = pd.DataFrame(columns = ["int_e_gain", "int_e_loss", "int_distance", "int_turns", "Longitude", "Latitude"])
		# iterate through results in each track
		for int_key, int_value in intervals.items():
			current_interval = split_dict[name][int_key]
			split_int_df = pd.DataFrame(columns = ["e_gain", "e_loss", "distance", 
										 "turns", 
										 "intE_gain", "intE_loss","intDistance", 
										 "intTurns"])
			for k, df in current_interval.items():
				split_int_df = split_int_df.append(df, ignore_index = True)
				temp_dict = {"int_e_gain": df['intE_gain'], "int_e_loss": df['intE_loss'], 
							 "int_distance": df['intDistance'], "int_turns": df['intTurns'], 
							 "Longitude": df["Longitude"], "Latitude": df["Latitude"]}
				predict_track_df = predict_track_df.append(temp_dict, ignore_index=True)

			X = split_int_df[['e_gain', 'e_loss', 'turns', 'distance', 'intDistance', 'intE_gain', 'intE_loss', 'intTurns']]
			split_int_distances = split_int_df['intDistance'].tolist()
			scaled_x = int_scale_x.transform(X)
			split_int_result = int_regr.predict(scaled_x)
			split_int_result = int_scale_y.inverse_transform(split_int_result)
			total_time = sum(split_int_result)
			# iterate through predicted results
			for r in range(len(split_int_result)):
				split_int_result[r] = split_int_result[r]/total_time
			interval_results.append(split_int_result)
			interval_distances.append(split_int_distances)

	interval_speeds = []
	for i in range(len(interval_results)):
		for j in range(len(interval_results[i])):
			interval_speeds.append(interval_distances[i][j]/(interval_results[i][j]*res_list[i]))
	totalTime = 0
	for res in res_list:
		totalTime += res

	dist_row = activities_df.iloc[-1]
	dist = dist_row.loc['distance']

	speed = dist/totalTime

	return totalTime, speed, interval_speeds, predict_track_df

		

if __name__ == "__main__":
	test_route_dir = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/testRoutes"
	results = predict_track(test_route_dir)
	print(results[0]/3600)
	print(results[1])
	print(results[2])
	print(results[3])
	#interval_speeds
	#for i in results[2]:
		#for i in range(10):


	map_df = results[3]
	print(map_df)


	pds = ProDataSimulator(path = test_route_dir, simulate = True)
	all_activities = pds.getAllActivities()
	for name, activity in all_activities.items():
		activity = pds.findAllMeasures(activity, Time = False, Speed = False, Slope = False, columnLabels = {'dist': "distance", 
																													   'e_gain': "e_gain", 
																													   'e_loss': "e_loss"})
	e_g = numpy.array(map_df["int_e_gain"])
	e_l = numpy.array(map_df["int_e_loss"])
	elev = e_g - e_l
	turns = map_df["int_turns"]
	int_dist = map_df["int_distance"]

	mapped_speed = []
	interval_length = round(len(activity) / len(results[2]))
	for i in range(len(results[2])):
		for j in range(interval_length):
			if i = 0:
				
			mapped_speed.append(speed)

	print(len(activity))
	print(len(mapped_speed))



	
	sns.set_theme()
	sns.set_context('notebook')
	fig = plt.figure()

	ax = fig.add_subplot(5,1,1)
	ax.set_ylim(0, 30)
	sns.lineplot(ax=ax, x=range(len(results[2])), y=results[2])

	ax1 = fig.add_subplot(5,1,2)
	sns.lineplot(ax=ax1, x=range(len(elev)), y=elev)

	ax2 = fig.add_subplot(5,1,3)
	sns.lineplot(ax=ax2, x=range(len(turns)), y=turns)

	ax3 = fig.add_subplot(5,1,4)
	sns.lineplot(ax=ax3, x=range(len(int_dist)), y=int_dist)

	fig2 = plt.figure()
	ax4 = fig2.add_subplot(1,1,1)
	sns.scatterplot(ax=ax4, data = activity, x='Longitude', y='Latitude')#, hue='Longitude', palette='flare')#, palette='flare')



	plt.show()




		



