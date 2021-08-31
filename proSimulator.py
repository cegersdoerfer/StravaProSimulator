from xml.dom.minidom import parse
import os
import pandas as pd
import traceback
import csv
import random
import math
from geopy import distance
import haversine
from datetime import datetime

class ProDataSimulator():
	def __init__(self, Path = None):
		self.Path = Path if not Path is None else "/Users/chris_egersdoerfer/Desktop/Strava-ProData"

	def getAllActivities(self):
		"""
			This method gets all activities of all athletes in the given strava data directory

			Return:
				activities: (dict) collection of all dataframes for each athlete where each 
				key is the name of an athlete and each value is the repective list of activities
		"""
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
		"""
			This method gets all activities of all athletes of a certain gender 

			Args:
				gender: (str) represents the desired gender for output

			Return:
				activities: (dict) collection of all dataframes for each athlete of the desired 
				gender where each key is the name of the athlete and each value is the respective list of activities

		"""
		if gender.lower() != "male" and gender.lower() != "female":
			print("Enter a valid gender (male/female)")
			return False
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
		"""
			This method parses all gpx files of a single athlete to individual pandas dataframes

			Args:
				Athlete: (str) represents the desired athlete

			Return: 
				dataframes: (list) collection of dataframes containing one for each activity of the given athlete
		"""
		athletePath = self.Path + '/' + athlete
		gpxFiles = os.listdir(athletePath)
		try:
			gpxFiles.remove(".DS_Store")
		except:
			pass
		parsed_gpx = self.parseGpx(athletePath, gpxFiles)
		dataframes = self.writeDataframes(parsed_gpx)

		return dataframes

	def getRandomActivitiesByGender(self, gender, sampleSize = 10):
		"""
			This method gets a random sample of activities of a certain gender

			Args:
				gender: (str) represents the desired gender
				sampleSize: (int) represents the amount of activities to be returned in the sample

			Return:
				selectedActivites: (list) collection of randomly selected dataframes of a specified gender
		"""
		summedActivities = {}
		athletes = os.listdir(self.Path)
		for athlete in athletes:
			if gender.lower() == "male" and "male" in athlete:
				activities = self.Path + '/' + athlete
				activities_list = os.listdir(activities)
				try:
					activities_list.remove(".DS_Store")
				except:
					pass
				for activity in activities_list:
					summedActivities[activity] = athlete
			elif gender.lower() == "female" and "female" in athlete:
				activities = self.Path + '/' + athlete
				activities_list = os.listdir(activities)
				try:
					activities_list.remove(".DS_Store")
				except:
					pass
				for activity in activities_list:
					summedActivities[activity] = athlete

		"""
		allGenderedActivities = self.getAllGenderedActivities(gender)
		for athlete in allGenderedActivities:
			for i in allGenderedActivities[athlete]:
				summedActivities.append(i)
		"""

		print(len(summedActivities))

		while sampleSize > len(summedActivities):
			print("Sample size larger than activities population. The Population is " + str(len(summedActivities)) + " activities.")
			sampleSize = int(input("Please enter a number lower than the population: "))
		
		selectedIndexes = random.sample(range(0, len(summedActivities)), sampleSize)
		
		selectedActivities = {}
		for index in selectedIndexes:
			selectedActivities[list(summedActivities.keys())[index]] = list(summedActivities.values())[index]


		#search for files using OS Module
		parsed_dict = {}
		for file in selectedActivities:
			parsed_gpx = self.parseGpx(self.Path + "/" + selectedActivities[file], [file])
			parsed_dict[file] = parsed_gpx[file]

		dataframes = self.writeDataframes(parsed_dict)

		return dataframes

	def getRandomActivitiesByAthlete(self, athlete, sampleSize = 1):
		"""
			This method gets a random sample of activities of a specified athlete

			Args:
				athlete: (str) name of the desired athlete
				sampleSize: (int) represents the amount of activities to be returned in the sample

			Return:
				selectedActivites: (list) collection of randomly selected dataframes of a specified athlete

		"""
		athletePath = self.Path + '/' + athlete
		allActivities = os.listdir(athletePath)
		allActivities.remove(".DS_Store")

		#allActivities = self.getSingleAthleteActivities(athlete)

		while sampleSize > len(allActivities):
			print("Sample size larger than activities population. The Population is " + str(len(summedActivities)) + " activities.")
			sampleSize = int(input("Please enter a number lower than the population: "))
		
		selectedIndexes = random.sample(range(0, len(allActivities)), sampleSize)

		selectedActivities =[]
		for index in selectedIndexes:
			selectedActivities.append(allActivities[index])

		parsed_gpx = self.parseGpx(athletePath, selectedActivities)
		dataframes = self.writeDataframes(parsed_gpx)

		return dataframes



	def parseGpx(self, dir, files = []):
		"""
			This method parses all activity files in the directory from gpx format to csv format

			Args:
				dir: (str) directory of the files to be parsed
				files: (list) a collection of files to be parsed in the specified directory

			Return:
				parsed_files: (dict) a collection of lists containing a separate dictionary for each datapoint. 
				Each list within the parsed_files list represents the datapoints of a separate file.
		"""
		if len(files) == 0:
			files = self.getAllActivites()

		parsed_files = {}
		for file in files:
			dataPoints = []
			with open(dir+"/"+file, 'r') as gpxFile:
				try:
					gpx = parse(gpxFile)
				except:
					print(traceback.format_exc())
					pass
				print("FILE " + dir+"/"+file)
				trackpoints = gpx.getElementsByTagName('trkpt')
				elevation = gpx.getElementsByTagName('ele')
				time = gpx.getElementsByTagName('time')
				for point in range(len(trackpoints)):
						dic = {"Time" : datetime.strptime(time[point].firstChild.nodeValue, '%Y-%m-%dT%H:%M:%S.%fZ'),
								"Latitude" : float(trackpoints[point].attributes["lat"].value),
								"Longitude" : float(trackpoints[point].attributes["lon"].value),
								"Elevation" : float(elevation[point].firstChild.nodeValue)
								}
						dataPoints.append(dic)
			parsed_files[file] = dataPoints

		return parsed_files

	def writeDataframes(self, parsed_dict):
		"""
			This method creates a collection of pandas dataframes from a list of files in the form returned by the parseGpx method

			Args:
				parsed_list: (list) a collection of lists where each list represents all datapoints of a strava gpx files

			Return:
				dataframes: (list) a collection of pandas dataframes, where each dataframe contains the datapoints of a separate gpx file

		"""
		dataframes = {}
		for data in parsed_dict:
			new_dataframe = pd.DataFrame(parsed_dict[data])
			dataframes[data] = new_dataframe

		return dataframes



	def findTotalDistance(self, dataframe, formula = "vincenty", includeElevation = True):
		dist = [0]
		for i in range(len(dataframe)):
			if i == 0:
				pass
			else:
				pStart = dataframe.iloc[i-1]
				pEnd = dataframe.iloc[i]

				if formula == "haversine":
					calculated_distance = haversine.haversine(
															(pStart.Latitude, pStart.Longitude), 
															(pEnd.Latitude, pEnd.Longitude)) * 1000
				elif formula == "vincenty":
					calculated_distance = distance.geodesic(
															(pStart.Latitude, pStart.Longitude), 
															(pEnd.Latitude, pEnd.Longitude)).m
				else:
					raise ValueError("invalid formula. Enter either 'vincenty' or 'haversine'"
									" for fomula parameter, or accept default value 'haversine'.")


				if includeElevation:
					calculated_distance = math.sqrt(calculated_distance**2 +(pEnd.Elevation - pStart.Elevation)**2)

				dist.append(dist[-1] + calculated_distance)

		if formula == "haversine":
			if includeElevation:
				dataframe["3dHavDistance"] = dist
			else:
				dataframe["2dHavDistance"] = dist
		else:
			if includeElevation:
				dataframe["3dVinDistance"] = dist
			else:
				dataframe["2dVinDistance"] = dist

		return dist[-1]



	def findTotalElevationGain(self, dataframe):
		elev = [0]

		for i in range(len(dataframe)):
			if i == 0:
				pass
			else:
				pStart = dataframe.iloc[i-1]
				pEnd = dataframe.iloc[i]

				elev_dif = pEnd.Elevation - pStart.Elevation

				if elev_dif > 0:
					elev.append(elev[-1] + elev_dif)
				else:
					elev.append(elev[-1])
		dataframe["elevationGain"] = elev

		return elev[-1]




	def findTotalElevationLoss(self, dataframe):
		elev = [0]

		for i in range(len(dataframe)):
			if i == 0:
				pass
			else:
				pStart = dataframe.iloc[i-1]
				pEnd = dataframe.iloc[i]

				elev_dif = pEnd.Elevation - pStart.Elevation

				if elev_dif < 0:
					elev.append(elev[-1] + abs(elev_dif))
				else:
					elev.append(elev[-1])
		dataframe["elevationLoss"] = elev

		return elev[-1]

	def findAverageSlope(self):
		pass

	def findSteepSections(self, sectionCount):
		"""
		returns a list of ones or zeroes to indicate where in the track 
		"""
		pass

	def findAvgSpeed(self, dataframe, distance = None, time = None):
		if distance == None:
			distance = self.findTotalDistance(dataframe)

		if time == None:
			time = self.findTotalTime(dataframe)

		return round(distance / time, 3)

	def findTotalTime(self, dataframe):
		"""
		finds the total time it took to finish the route


		"""
		startTime = dataframe.iloc[1].Time
		endTime = dataframe.iloc[-1].Time

		totalTime = round((endTime - startTime).total_seconds()/60, 2)

		return totalTime



	def findTurns(self, points, threshold = 20, minLength = 1):
		"""
		This method calculates the amount of turns and the total degree of each turn from a list of coordinates

		Args:
			points: (list) a collection of coordinates in the form [x,y]
			threshold: (int) the minimum degree change between the course of one point and the next to signify a turn
			minLength: (int) the minimum length in points to classify as a turn (default of 1 evaluates to 3 so actual 
						amount of points included is equal to (minLength * 2) + 1)

		return:
			turns: (list) a collection of turns where each turn is it's own list of points (not including the first point).
					Each point in each turn is followed by the calculated course of that point and the last value in the 
					list of each turn is the total degree of the entire turn.
		"""
		turnStart = 0
		turnLength = 0
		prevAngleDiff = 0
		turns = []
		
		i = 0
		while i < len(points)-1:
			turns.append([])
			sumAngle = 0
			if i < 2:
				pass
			else:
				startPoint = i-2
				newPointStart = i-1
				newPointEnd = i
				referenceAngle = self.findCourse(points[startPoint], points[newPointStart])
				newAngle = self.findCourse(points[newPointStart], points[newPointEnd])
				angle_diff = newAngle - referenceAngle
				sumAngle += angle_diff
				if(angle_diff > threshold):
					while angle_diff > threshold:
						turns[-1].append(points[newPointEnd])
						turns[-1].append(newAngle)
						try:
							newPointStart += 1
							newPointEnd += 1
							newAngle = self.findCourse(points[newPointStart], points[newPointEnd])
							angle_diff = newAngle - turns[-1][-1]
							if newAngle < 90 and turns[-1][-1] > 270:
								angle_diff += 360
							sumAngle += angle_diff
							if angle_diff < threshold:
								i = newPointEnd
								turns[-1].append(sumAngle)

						except IndexError:
							i = len(points)

				elif angle_diff < (-1 * threshold):
					while angle_diff < (-1 * threshold):
						turns[-1].append(points[newPointEnd])
						turns[-1].append(newAngle)
						try:
							newPointStart += 1
							newPointEnd += 1
							newAngle = self.findCourse(points[newPointStart], points[newPointEnd])
							angle_diff = newAngle - turns[-1][-1]
							if newAngle > 270 and turns[-1][-1] < 90:
								angle_diff -= 360
							sumAngle += angle_diff
							if angle_diff > (-1*threshold):
								i = newPointEnd
								turns[-1].append(sumAngle)


						except IndexError:
							i = len(points)
			if turns[-1] == [] or len(turns[-1]) <= (minLength * 2 + 1):
				del turns[-1]
			i+=1
		return turns




	def findCourse(self, point1, point2):
		"""
		This method calculates the angle between 2 point (also known as the course)

		Args:
			point1: (list) coordinate of the first point in the form [x,y]
			point2: (list) coordinate of the second point in the form [x,y]

		Return:
			course: (float) the angle in degrees between the two given points 
		"""
		convertFactor = 180 / math.pi	
		x_diff = point2[0] - point1[0]
		y_diff = point2[1] - point1[1]
		angle = math.atan2(y_diff , x_diff)
		if angle < 0:
			angle = angle + (2 * math.pi)
		course = angle * convertFactor
		return course

	def findEstimatedExertion(self):
		pass

	def findFastestInterval(self):
		pass

	def findIntervalAvgSpeed(self):
		pass

	def findIntervalElevationGain(self):
		pass

	def findIntervalsByNum(self, track, intervalCount = 3):
		"""

		"""
		interval_length = len(track) // intervalCount
		intervals = {}
		for i in range(intervalCount):
			if i < (intervalCount-1):
				intervals["interval " + str(i)] = track.iloc[interval_length * i : interval_length * (i+1), :]
			else:
				print("last interval")
				intervals["interval " + str(i)] = track.iloc[interval_length * i:, :]
		return intervals


	def FindFastestSetLengthInterval(self, track, intervalLength = 10000, formula = "vincenty", includeElevation = True):
		"""

		"""
		intervalStart = 0
		pointCount = 0
		shortestInterval = 0
		for i in range(len(track)):
			if i == 0:
				pass
			else:
				pStart = dataframe.iloc[i-1]
				pEnd = dataframe.iloc[i]

				if distanceFormula == "haversine":
					calculated_distance = haversine.haversine(
															(pStart.Latitude, pStart.Longitude), 
															(pEnd.Latitude, pEnd.Longitude)) * 1000
				elif formula == "vincenty":
					calculated_distance = distance.geodesic(
															(pStart.Latitude, pStart.Longitude), 
															(pEnd.Latitude, pEnd.Longitude)).m
				else:
					raise ValueError("invalid formula. Enter either 'vincenty' or 'haversine'"
									"for fomula parameter, or accept default value 'haversine'.")


				if includeElevation:
					calculated_distance = math.sqrt(calculated_distance**2 +(pEnd.Elevation - pStart.Elevation)**2)

				dist.append(dist[-1] + calculated_distance)

				if dist[-1] > intervalLength:
					currentInterval = track.iloc[intervalStart:i+1, :]
					#if len(currentInterval)





		
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

if __name__ == '__main__':

	test = ProDataSimulator()
	#test.getRandomActivitiesByGender("female")
	#print(len(test.getAllActivities()))
	#print(len(test.getAllGenderedActivities("female")))

	#activity = test.getRandomActivitiesByGender("female", sampleSize = 1)
	activity = test.getSingleAthleteActivities("chris - male")

	#activity = test.getRandomActivitiesByAthlete("Lucina Brand - female", sampleSize = 2)
	for i in activity:
		print(activity[i])
		dis = test.findTotalDistance(activity[i])
		"""
		print("haversine 2d distance: " + str(test.findTotalDistance(activity[i], includeElevation = False)))
		print("haversine 3d distance: " + str(test.findTotalDistance(activity[i])))
		print("vincenty 2d distance: " + str(test.findTotalDistance(activity[i], formula = "vincenty", includeElevation = False)))
		print("vincenty 3d distance: " + str(test.findTotalDistance(activity[i], formula = "vincenty")))
		"""
		gain = test.findTotalElevationGain(activity[i])
		print("elevation gain: " + str(gain))
		loss = test.findTotalElevationLoss(activity[i])
		print("elevation loss: " + str(loss))
		print("elevation difference: " + str(gain - loss))

		intervals = test.findIntervalsByNum(activity[i], 5)
		print(intervals)
		print(intervals["interval 4"].iloc[-1,:])
		print(activity[i].iloc[-1,:])
		sumLength = 0
		for interval in intervals:
			sumLength += len(intervals[interval])

		print("intervals Length: " + str(sumLength))
		print("dataframe Length: " + str(len(activity[i])))

		print(test.findTotalTime(activity[i]))

		points = []
		for r in range(len(activity[i])):
			row = activity[i].iloc[r]
			points.append([row.Longitude, row.Latitude])
		turns = test.findTurns(points, threshold = 20, minLength = 1)
		print("amount of turns: " + str(len(turns)))


		s = 0
		degree = 0
		for turn in turns:
			s += len(turn)
			degree += abs(turn[-1])
		avgDegree = degree/len(turns)
		avgLength = s/len(turns)
		print("average length of turns: " + str(avgLength))
		print("average degree of turns: " + str(avgDegree))
		print("average distance of points: " + str(dis / len(activity[i])))




		print("FINISHED CALCULATIONS FOR: " + str(i))
		print("\n" + "--------------------------------------" + "\n")





