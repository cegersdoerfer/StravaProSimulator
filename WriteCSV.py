import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator
import random


#all_activities = {}
#activities = pds.getSingleAthleteActivities("chris - male")
#all_activities["chris - male"] = activities
#all_activities = [i for i in [value for key, value in all_activities.items()]]
#for name, activity in all_activities.items():
	#activity.to_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/" + name, index = False)

#all_male_activities = pds.getAllGenderedActivities("male")
#all_female_activities = pds.getAllGenderedActivities("female")
def create_csv(all_activities, simulator, simulate = False, intervals = False, filePath = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/proData-csv/test"):
	if simulate is False:
		if intervals is False:

			measure_df = pd.DataFrame(columns = ["time", "e_gain", "e_loss", "distance", "avgSpeed", 
												 "turns"])
		else:

			measure_df = pd.DataFrame(columns = ["time", "e_gain", "e_loss", "distance", "avgSpeed", 
												 "turns", "intTime", 
												 "intE_gain", "intE_loss","intDistance", "intAvgSpeed", 
												 "intTurns"])
	else:
		if intervals is False:

			measure_df = pd.DataFrame(columns = ["e_gain", "e_loss", "distance", 
												 "turns"])
		else:

			measure_df = pd.DataFrame(columns = ["e_gain", "e_loss", "distance", 
												 "turns", 
												 "intE_gain", "intE_loss","intDistance", 
												 "intTurns"])
	progress = 0
	for name, activity in all_activities.items():
		if simulate is False:
			activity = simulator.findAllMeasures(activity, columnLabels = {'dist': "distance", 
																		   'e_gain': "e_gain", 
																		   'e_loss': "e_loss", 
																		   'time': "time", 
																		   'speed': "speed", 
																		   'slope': "slope"})
		else:
			activity = simulator.findAllMeasures(activity, Time = False, Speed = False, columnLabels = {'dist': "distance", 
																										'e_gain': "e_gain", 
																										'e_loss': "e_loss", 
																										'slope': "slope"})
		points = []
		for r in range(len(activity)):
				row = activity.iloc[r]
				points.append([row.Longitude, row.Latitude])
		turns = simulator.findTurns(points)
		print(len(turns[1]))

		if simulate is False:
			avgSpeed = sum(activity["speed"])/len(activity)

		preceding_slopes = simulator.findPrecedingSlope(turns[1], activity)
		shift = abs(min(preceding_slopes))
		preceding_slopes = [(shift + i) for i in preceding_slopes]
		maxSlope = max(preceding_slopes)
		minSlope = min(preceding_slopes)
		normalized_slopes = []
		up_counter = 0
		for slope in preceding_slopes:
			normalized = slope/(maxSlope - minSlope)
			if normalized > 0.4:
				up_counter += 1
			normalized_slopes.append(normalized)
		#print("up counter: " + str(up_counter))
		#print("normalized: " + str(len(normalized_slopes)))




		lastRow = activity.iloc[-1]

		if intervals is False:
			print(name, " : ", lastRow)
			print("iteration: " + str(progress))
			if simulate is False:
				lastRow = lastRow[["time", "e_gain", "e_loss", "distance", 'turns']]
				lastRow["avgSpeed"] = avgSpeed
			else:
				lastRow = lastRow[["e_gain", "e_loss", "distance", 'turns']]
			lastRow["downHillTurns"] = len(turns[0]) - up_counter
			lastRow["turns"] = len(turns[0])

			measure_df = measure_df.append(lastRow, ignore_index = True)

		else:

			activityIntervals = simulator.findIntervalsByNum(activity, intervalCount = random.randrange(2,5))
			activityLastRow = lastRow
			lastRow["turns"] = len(turns[0])
			lastRow["downHillTurns"] = len(turns[0]) - up_counter
			if simulate is False:
				activityLastRow = activityLastRow[["time", "e_gain", "e_loss", "distance", "turns", "downHillTurns"]]
			else:
				activityLastRow = activityLastRow[["e_gain", "e_loss", "distance", "turns", "downHillTurns"]]
			print(activityLastRow)
			for interval in activityIntervals:
				if simulate is False:
					interval = simulator.findAllMeasures(activityIntervals[interval], columnLabels = {'dist': "intDistance", 
																									  'e_gain': "intE_gain", 
																									  'e_loss': "intE_loss", 
																									  'time': "intTime", 
																									  'speed': "intSpeed", 
																									  'slope': "intSlope"})
				else:
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
				
				if simulate is False:
					avgSpeed = sum(interval["speed"])/len(interval)

				preceding_slopes = simulator.findPrecedingSlope(turns[1], interval)
				try:
					shift = abs(min(preceding_slopes))
					preceding_slopes = [(shift + i) for i in preceding_slopes]
					maxSlope = max(preceding_slopes)
					minSlope = min(preceding_slopes)
				except:
					preceding_slopes = []
				normalized_slopes = []
				up_counter = 0
				for slope in preceding_slopes:
					try:
						normalized = slope/(maxSlope - minSlope)
					except:
						normalized = 0
					if normalized > 0.4:
						up_counter += 1
					normalized_slopes.append(normalized)

				lastRow = interval.iloc[-1,:]
				
				if simulate is False:
					lastRow = lastRow[["intTime", "intE_gain", "intE_loss", "intDistance"]]
					lastRow.loc["time"] = activityLastRow["time"]
					lastRow["avgSpeed"] = avgSpeed
				else:
					lastRow = lastRow[["intE_gain", "intE_loss", "intDistance"]]
				lastRow.loc["e_gain"] = activityLastRow["e_gain"]
				lastRow.loc["e_loss"] = activityLastRow["e_loss"]
				lastRow.loc["distance"] = activityLastRow["distance"]
				lastRow.loc["turns"] = activityLastRow["turns"]
				lastRow.loc["downHillTurns"] = activityLastRow["downHillTurns"]
				lastRow["intDownHillTurns"] = len(turns[0]) - up_counter
				lastRow["intTurns"] = len(turns[0])
				print(name, " : ", lastRow)
				print("iteration: " + str(progress))

				measure_df = measure_df.append(lastRow, ignore_index = True)

		progress += 1
	
	print(measure_df)


	measure_df.to_csv(filePath, index = False)

	return measure_df


if __name__ == '__main__':
	pds = ProDataSimulator("/Users/chris_egersdoerfer/Desktop/Strava-ProData")

	all_activities = pds.getRandomActivitiesByGender("male", 158)
	activities_df = create_csv(all_activities, pds, intervals = True, filePath = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/proData-csv/test_all_male_intervals_2-40")







