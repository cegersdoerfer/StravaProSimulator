import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator
import random
import sys
import json





#all_activities = {}
#activities = pds.getSingleAthleteActivities("chris - male")
#all_activities["chris - male"] = activities
#all_activities = [i for i in [value for key, value in all_activities.items()]]
#for name, activity in all_activities.items():
	#activity.to_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/" + name, index = False)

#all_male_activities = pds.getAllGenderedActivities("male")
#all_female_activities = pds.getAllGenderedActivities("female")
def create_csv(all_activities, simulator, random_range = [2, 40], csv = True, filePath = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/proData-csv/test"):
	measure_df = pd.DataFrame(columns = ["time", "e_gain", "e_loss", "distance", 
										 "turns", "intTime", 
										 "intE_gain", "intE_loss","intDistance", 
										 "intTurns"])
	prog = 0
	total_length = len(all_activities.items())
	simulator.progress(prog, total_length, 'calculating features from gpx')
	interval_dict = {}
	for name, activity in all_activities.items():
		activity = simulator.findAllMeasures(activity, columnLabels = {'dist': "distance", 
																	   'e_gain': "e_gain", 
																	   'e_loss': "e_loss", 
																	   'time': "time", 
																	   'speed': "speed", 
																	   'slope': "slope"})
		points = []
		for r in range(len(activity)):
				row = activity.iloc[r]
				points.append([row.Longitude, row.Latitude])
		turns = simulator.findTurns(points)

		avgSpeed = sum(activity["speed"])/len(activity)

		"""
		preceding_slopes = simulator.findPrecedingSlope(turns[1], activity)
		shift = abs(min(preceding_slopes))
		preceding_slopes = [(shift + i) for i in preceding_slopes]
		maxSlope = max(preceding_slopes)
		minSlope = min(preceding_slopes)
		normalized_slopes = []
		up_counter = 0
		for slope in preceding_slopes:
			normalized = (slope - minSlope)/(maxSlope - minSlope)
			if normalized > 0.4:
				up_counter += 1
			normalized_slopes.append(normalized)
		"""
		lastRow = activity.iloc[-1, :]
		activityIntervals = simulator.findIntervalsByNum(activity, intervalCount = random.randint(random_range[0],random_range[1]))
		activityLastRow = lastRow.copy()
		activityLastRow.loc["turns"] = len(turns[0])
		#activityLastRow.loc["downHillTurns"] = len(turns[0]) - up_counter
		activityLastRow = activityLastRow[["time", "e_gain", "e_loss", "distance", "turns"]]
		interval_dict.update({name: {}})
		for interval_name in activityIntervals:
			interval = simulator.findAllMeasures(activityIntervals[interval_name], columnLabels = {'dist': "intDistance", 
																							  'e_gain': "intE_gain", 
																							  'e_loss': "intE_loss", 
																							  'time': "intTime", 
																							  'speed': "intSpeed", 
																							  'slope': "intSlope"})
			if csv is False:
				interval_dict[name][interval_name] = interval.to_json()
			else:
				interval_dict[name][interval_name] = interval
			points = []
			for r in range(len(interval)):
				row = interval.iloc[r]
				points.append([row.Longitude, row.Latitude])
			turns = simulator.findTurns(points)
				

			avgSpeed = sum(interval["speed"])/len(interval)

			lastRow = interval.iloc[-1,:]	
			lastRow = lastRow[["intTime", "intE_gain", "intE_loss", "intDistance"]]
			lastRow.loc["time"] = activityLastRow["time"]
			lastRow.loc["avgSpeed"] = avgSpeed
			lastRow.loc["e_gain"] = activityLastRow["e_gain"]
			lastRow.loc["e_loss"] = activityLastRow["e_loss"]
			lastRow.loc["distance"] = activityLastRow["distance"]
			lastRow.loc["turns"] = activityLastRow["turns"]
			#lastRow.loc["downHillTurns"] = activityLastRow["downHillTurns"]
			#lastRow.loc["intDownHillTurns"] = len(turns[0]) - up_counter
			lastRow.loc["intTurns"] = len(turns[0])

			simulator.progress(prog, total_length, 'calculating features from gpx')
			measure_df = measure_df.append(lastRow, ignore_index = True)

		prog += 1

	if csv == True:
		measure_df.to_csv(filePath, index = False)
		return interval_dict 
	else:
		return measure_df


if __name__ == '__main__':
	pds = ProDataSimulator("/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/Strava-ProData")

	all_activities = pds.getRandomActivitiesByGender("male", 1)
	print("\n")
	print("GPX PARSING COMPLETE...")
	print("--------------------------------------" + "\n")
	print("Analyzing gpx files for features...this may take a while")
	filepath = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/proData-csv/test_all_male_intervals_2-40Test"
	activities_df = create_csv(all_activities, pds, filePath = filepath)
	print("\n")
	print("FEATURE CALCULATION COMPLETE...")
	print("--------------------------------------" + "\n")
	print("csv with features can be found at: " + filepath)
	print("\n")
	print("Breaking intervals and analyzing for features...this may really take a while")
	count = len(activities_df)
	curCount = 0
	split_dict = {}
	for name in activities_df:
		curCount += 1
		print("analyzing: " + name + "...")
		print("Progress: " + str(curCount) + " out of " + str(count))
		broken_activities = create_csv(activities_df[name], pds, random_range = [2, 20], csv = False, filePath = filepath+"P2")
		split_dict[name] = broken_activities
		print("\n")
		

	print(split_dict)











