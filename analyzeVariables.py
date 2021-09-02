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
def create_csv(all_activities, filePath = "/Users/chris_egersdoerfer/Desktop/proData-csv/test"):
	measure_df = pd.DataFrame(columns = ["time", "e_gain", "e_loss", "distance", "avgSpeed", "turns", "avgTurnDegree", "avgTurnLength"])
	for name, activity in all_activities.items():
		activity = pds.findAllMeasures(activity, columnLabels = ["distance", "e_gain", "e_loss", "time", "speed"])

		points = []
		for r in range(len(activity)):
				row = activity.iloc[r]
				points.append([row.Longitude, row.Latitude])
		turns = pds.findTurns(points)

		s = 0
		degree = 0
		for turn in turns:
				s += len(turn)
				degree += abs(turn[-1])

		avgDegree = degree/len(turns)
		avgLength = s/len(turns)
		avgSpeed = sum(activity["speed"])/len(activity)

		lastRow = activity.iloc[-1]
		print(lastRow)
		lastRow = lastRow[["time", "e_gain", "e_loss", "distance"]]
		lastRow["avgSpeed"] = avgSpeed
		lastRow["turns"] = len(turns)
		lastRow["avgTurnDegree"] = avgDegree
		lastRow["avgTurnLength"] = avgLength


		measure_df = measure_df.append(lastRow, ignore_index = True)

	print(measure_df)

	measure_df.to_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/test", index = False)

	return measure_df



pds = ProDataSimulator("/Users/chris_egersdoerfer/Desktop/Strava-ProData")

all_activities = pds.getRandomActivitiesByGender("male", 25)
activities_df = create_csv(all_activities, "/Users/chris_egersdoerfer/Desktop/proData-csv/test_male_30")

fig = plt.figure()

ax1 = fig.add_subplot(331)
ax1.scatter(activities_df['e_gain'], activities_df['time']/60)
#ax1.scatter(activities_df['e_gain'], activities_df['avgSpeed']**2)


ax2 = fig.add_subplot(332)
ax2.scatter(activities_df['e_loss'], activities_df['time']/60)
#ax2.scatter(activities_df['e_loss'], activities_df['avgSpeed']**2)


ax3 = fig.add_subplot(333)
ax3.scatter(activities_df['distance'], activities_df['time']/60)
#ax3.scatter(activities_df['distance'], activities_df['avgSpeed']**2)


ax4 = fig.add_subplot(334)
ax4.scatter(activities_df['avgSpeed'], activities_df['time']/60)
#ax4.scatter(activities_df['avgSpeed'], activities_df['avgSpeed']**2)


ax5 = fig.add_subplot(335)
ax5.scatter(activities_df['turns'], activities_df['time']/60)
#ax5.scatter(activities_df['turns'], activities_df['avgSpeed']**2)


ax6 = fig.add_subplot(336)
ax6.scatter(activities_df['avgTurnDegree'], activities_df['time']/60)
#ax6.scatter(activities_df['avgTurnDegree'], activities_df['avgSpeed']**2)


ax7 = fig.add_subplot(337)
ax7.scatter(activities_df['avgTurnLength'], activities_df['time']/60)
#ax7.scatter(activities_df['avgTurnLength'], activities_df['avgSpeed']**2)

ax8 = fig.add_subplot(338)
ax8.scatter(activities_df['e_gain']/activities_df['distance'], activities_df['time']/60)


plt.show()




