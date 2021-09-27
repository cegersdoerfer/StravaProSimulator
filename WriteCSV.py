import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator


#all_activities = {}
#activities = pds.getSingleAthleteActivities("chris - male")
#all_activities["chris - male"] = activities
#all_activities = [i for i in [value for key, value in all_activities.items()]]
#for name, activity in all_activities.items():
	#activity.to_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/" + name, index = False)

#all_male_activities = pds.getAllGenderedActivities("male")
#all_female_activities = pds.getAllGenderedActivities("female")
def create_csv(all_activities, simulator, intervals = False, filePath = "/Users/chris_egersdoerfer/Desktop/proData-csv/test"):
	if intervals is False:

		measure_df = pd.DataFrame(columns = ["time", "e_gain", "e_loss", "distance", "avgSpeed", 
											 "turns", "avgTurnDegree", "avgTurnLength"])
	else:

		measure_df = pd.DataFrame(columns = ["time", "e_gain", "e_loss", "distance", "avgSpeed", 
											 "turns", "avgTurnDegree", "avgTurnLength", "intTime", 
											 "intE_gain", "intE_loss","intDistance", "intAvgSpeed", 
											 "intTurns", "intAvgTurnDegree", "intAvgTurnLength"])
	progress = 0
	for name, activity in all_activities.items():
		activity = simulator.findAllMeasures(activity, columnLabels = ["distance", "e_gain", "e_loss", "time", "speed", "slope"])
		points = []
		for r in range(len(activity)):
				row = activity.iloc[r]
				points.append([row.Longitude, row.Latitude])
		turns = simulator.findTurns(points)
		print(len(turns[1]))

		s = 0
		degree = 0
		for turn in turns[0]:
				s += len(turn)
				degree += abs(turn[-1])

		avgDegree = degree/len(turns[0])
		avgLength = s/len(turns[0])
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





		if intervals is False:
			lastRow = activity.iloc[-1]
			print(name, " : ", lastRow)
			print("iteration: " + str(progress))
			lastRow = lastRow[["time", "e_gain", "e_loss", "distance"]]
			lastRow["avgSpeed"] = avgSpeed
			lastRow["downHillTurns"] = len(turns[0]) - up_counter
			lastRow["turns"] = len(turns[0])
			lastRow["avgTurnDegree"] = avgDegree
			lastRow["avgTurnLength"] = avgLength

			measure_df = measure_df.append(lastRow, ignore_index = True)

		else:

			activityIntervals = simulator.findIntervalsByNum(activity, intervalCount = 10)
			activityLastRow = activity.iloc[-1]
			activtyLastRow = activityLastRow[["time", "e_gain", "e_loss", "distance"]]
			print(activtyLastRow)
			for interval in activityIntervals:
				interval = simulator.findAllMeasures(activityIntervals[interval], columnLabels = ["intDistance", "intE_gain", "intE_loss", "intTime", "intSpeed", "intSlope"])
				points = []
				for r in range(len(interval)):
					row = activity.iloc[r]
					points.append([row.Longitude, row.Latitude])
				turns = simulator.findTurns(points)


				s = 0
				degree = 0
				for turn in turns[0]:
						s += len(turn)
						degree += abs(turn[-1])

				try:
					avgDegree = degree/len(turns[0])
				except:
					avgDegree = 0
				try:
					avgLength = s/len(turns[0])
				except:
					avgLength = 0
				
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
				
				lastRow = lastRow[["intTime", "intE_gain", "intE_loss", "intDistance"]]
				lastRow.loc["time"] = activtyLastRow["time"]
				lastRow.loc["e_gain"] = activtyLastRow["e_gain"]
				lastRow.loc["e_loss"] = activtyLastRow["e_loss"]
				lastRow.loc["distance"] = activtyLastRow["distance"]
				lastRow["avgSpeed"] = avgSpeed
				lastRow["downHillTurns"] = len(turns[0]) - up_counter
				lastRow["turns"] = len(turns[0])
				lastRow["avgTurnDegree"] = avgDegree
				lastRow["avgTurnLength"] = avgLength
				print(name, " : ", lastRow)
				print("iteration: " + str(progress))

				measure_df = measure_df.append(lastRow, ignore_index = True)

		progress += 1
	
	print(measure_df)


	measure_df.to_csv(filePath, index = False)

	return measure_df



pds = ProDataSimulator("/Users/chris_egersdoerfer/Desktop/Strava-ProData")

all_activities = pds.getRandomActivitiesByGender("male", 50)
activities_df = create_csv(all_activities, pds, intervals = True, filePath = "/Users/chris_egersdoerfer/Desktop/proData-csv/test_all_male_intervals")







