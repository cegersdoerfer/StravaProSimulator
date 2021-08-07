from xml.dom.minidom import parse
import os
import pandas as pd
import traceback
import csv
import random

class proDataImporter():
	def __init__(self, Path = None):
		self.Path = Path if not Path is None else "/Users/chris_egersdoerfer/Desktop/Strava-ProData"

	def getAllActivities(self):
		"""This method returns all activities in all athlete folders in the pro data directory"""
		activities = {}
		athletes = os.listdir(self.Path)
		athletes.remove(".DS_Store")
		for athlete in athletes:
			try:
				activities[athlete] = self.getSingleAthleteActivities(athlete)
			except:
				print(traceback.format_exc())
				pass

		return activities

	def getAllGenderedActivities(self, gender):
		if gender.lower() != "male" and gender.lower() != "female":
			print("Enter a valid gender (male/female)")
		genderSortedFiles = []
		athletes = os.listdir(self.Path)
		for athlete in athletes:
			if gender.lower() == "male" and "male" in athlete:
				genderSortedFiles.append(athlete)
			elif gender.lower() == "female" and "female" in athlete:
				genderSortedFiles.append(athlete)

		activities = {}
		for athlete in genderSortedFiles:
			activities[athlete] = self.getSingleAthleteActivities(athlete)

		return activities




	def getSingleAthleteActivities(self, athlete):
		athletePath = self.Path + '/' + athlete
		gpxFiles = os.listdir(athletePath)
		try:
			gpxFiles.remove(".DS_Store")
		except:
			pass
		parsed_gpx = self.parseGpx(athletePath, gpxFiles)
		dataframes = self.writeDataframes(parsed_gpx)
		print(dataframes)

		return dataframes

	def getRandomActivitiesByGender(self, gender, sampleSize = 10):
		allGenderedActivities = self.getAllGenderedActivities(gender)
		summedActivities = []
		for athlete in allGenderedActivities:
			for i in allGenderedActivities[athlete]:
				summedActivities.append(i)

		print(len(summedActivities))

		if sampleSize > len(summedActivities):
			print("Sample size larger than activities population")
		else:
			selectedIndexes = random.sample(range(0, len(summedActivities)), sampleSize)
		
		selectedActivities =[]
		for index in selectedIndexes:
			selectedActivities.append(allGenderedActivities[index])
		
		return selectedActivities

	def getRandomActivityByAthlete(self, athlete):
		pass

	def parseGpx(self, dir, files = []):
		if len(files) == 0:
			files = self.getAllActivites()

		parsed_files = []
		for file in files:
			dataPoints = []
			with open(dir+"/"+file, 'r') as gpxFile:
				try:
					gpx = parse(gpxFile)
				except:
					pass
				print("FILE " + dir+"/"+file)
				trackpoints = gpx.getElementsByTagName('trkpt')
				elevation = gpx.getElementsByTagName('ele')
				time = gpx.getElementsByTagName('time')
				for point in range(len(trackpoints)):
						dic = {"Time" : time[point].firstChild.nodeValue,
								"Latitude" : trackpoints[point].attributes["lat"].value,
								"Longitude" : trackpoints[point].attributes["lon"].value,
								"Elevation" : elevation[point].firstChild.nodeValue
								}
						dataPoints.append(dic)
			parsed_files.append(dataPoints)

		return parsed_files

	def writeDataframes(self, parsed_list):
		dataframes = []
		for data in parsed_list:
			new_dataframe = pd.DataFrame(data)
			dataframes.append(new_dataframe)

		return dataframes

	"""
	def writeCSV(self, parsed_list, athlete):
		for data in parsed_list:
			f = open("sample.csv", "w")
			writer = csv.DictWriter(
			    f, fieldnames=["fruit", "count"])
			writer.writeheader()
			writer.writerows(
			    [{"fruit": "apple", "count": "1"},
			    {"fruit": "banana", "count": "2"}])
			f.close()

	"""


#gpx = gpxpy.parse("/Users/chris_egersdoerfer/Desktop/Strava-ProData/Alex Dowsett - male/strava.activities.2963997118.Morning-Ride.gpx")
#print(gpx)


#os.chdir("/Users/chris_egersdoerfer/Downloads")
#dom1 = parse("strava.activities.5642665960.Afternoon-Ride.gpx")
#track = dom1.getElementsByTagName('time')
#print(track[1].firstChild.nodeValue)
#n_track = len(track)
#print(n_track)

#os.chdir("/Users/chris_egersdoerfer/Desktop/Strava-ProData/Alex Dowsett - male")

#basicname, file_extension = os.path.splitext("strava.activities.2963997118.Morning-Ride.gpx")

    #print(basicname,file_extension)
#s = open("Afternoon_Ride.gpx", mode='r', encoding='utf-8-sig').read()
#open(f"{basicname}_NOBOM{file_extension}", mode='w', encoding='utf-8').write(s)

#gpxpy.parse("Afternoon_Ride_NOBOM.gpx")


test = proDataImporter()
#print(len(test.getAllActivities()))
#print(len(test.getAllGenderedActivities("female")))
print(len(test.getRandomActivitiesByGender("female")))


