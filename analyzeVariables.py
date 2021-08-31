import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator

pds = ProDataSimulator("/Users/chris_egersdoerfer/Desktop/Strava-ProData")

all_activities = pds.getRandomActivitiesByGender("female", 30)
#all_activities = {}
#activities = pds.getSingleAthleteActivities("chris - male")
#all_activities["chris - male"] = activities
#all_activities = [i for i in [value for key, value in all_activities.items()]]
print(all_activities)
for name, activity in all_activities.items():
	activity.to_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/" + name, index = False)

#all_male_activities = pds.getAllGenderedActivities("male")
#all_female_activities = pds.getAllGenderedActivities("female")


