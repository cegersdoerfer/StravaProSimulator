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
import sys

class ProDataSimulator():
	def __init__(self, path = "/Users/chris_egersdoerfer/Documents/GitHub/StravaProSimulator/Strava-ProData", simulate = False):
		self.Path = path
		self.simulate = simulate

	def getPath(self):
		return self.path

	def setPath(self, path):
		self.path = path

	def getAllActivities(self):
		"""
			This method gets all activities of all athletes in the given strava data directory

			Return:
				activities: (dict) collection of all dataframes for each athlete where each 
				key is the name of an athlete and each value is the repective list of activities
		"""
		activities = {}
		if self.simulate is False:
			athletes = os.listdir(self.Path)
			athletes.remove(".DS_Store")
			if self.simulate is False:
				for athlete in athletes:
					try:
						activities[athlete] = self.getSingleAthleteActivities(athlete)
					except:
						print(traceback.format_exc())
						pass
		else:
			tracks = os.listdir(self.Path)
			tracks.remove(".DS_Store")
			for track in tracks:
				try:
					activities = self.getSingleAthleteActivities()
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




	def getSingleAthleteActivities(self, athlete = None):
		"""
			This method parses all gpx files of a single athlete to individual pandas dataframes

			Args:
				Athlete: (str) represents the desired athlete

			Return: 
				dataframes: (list) collection of dataframes containing one for each activity of the given athlete
		"""
		if athlete != None:
			dirPath = self.Path + '/' + athlete
		else:
			dirPath = self.Path
		gpxFiles = os.listdir(dirPath)
		try:
			gpxFiles.remove(".DS_Store")
		except:
			pass
		parsed_gpx = self.parseGpx(dirPath, gpxFiles)
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
			if "female" in athlete:
				if gender.lower() == "female":
					activities = self.Path + '/' + athlete
					activities_list = os.listdir(activities)
					try:
						activities_list.remove(".DS_Store")
					except:
						pass
					for activity in activities_list:
						summedActivities[activity] = athlete
			elif "male" in athlete:
				if gender.lower() == "male":
					activities = self.Path + '/' + athlete
					activities_list = os.listdir(activities)
					try:
						activities_list.remove(".DS_Store")
					except:
						pass
					for activity in activities_list:
							summedActivities[activity] = athlete

		while sampleSize > len(summedActivities):
			print("Sample size larger than activities population. The Population is " + str(len(summedActivities)) + " activities.")
			sampleSize = int(input("Please enter a number lower than the population: "))
		
		selectedIndexes = random.sample(range(0, len(summedActivities)), sampleSize)
		
		selectedActivities = {}
		for index in selectedIndexes:
			selectedActivities[list(summedActivities.keys())[index]] = list(summedActivities.values())[index]


		#search for files using OS Module
		parsed_dict = {}
		curCount = 0
		totalCount = len(selectedActivities)
		for file in selectedActivities:
			parsed_gpx = self.parseGpx(self.Path + "/" + selectedActivities[file], [file])
			parsed_dict[file] = parsed_gpx[file]
			curCount += 1
			self.progress(curCount, totalCount, 'reading gpx files')
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
				#print("FILE " + dir+"/"+file)
				if self.simulate is False:
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
				else:
					trackpoints = gpx.getElementsByTagName('trkpt')
					elevation = gpx.getElementsByTagName('ele')
					for point in range(len(trackpoints)):
							dic = {"Latitude" : float(trackpoints[point].attributes["lat"].value),
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

	def findAllMeasures(self, dataframe, formula = "vincenty", 
						includeElevation = True, Distance = True, 
						ElevationGain = True, ElevationLoss = True, 
						Time = True, Speed = True, Slope = True,
						columnLabels = None):
		dist = [0]
		e_gain = [0]
		e_loss = [0]
		time_measure = [0]
		speed_measure = [0]
		slope_measure = [0]

		
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

				elev_dif = pEnd.Elevation - pStart.Elevation

				if ElevationGain:
					if elev_dif > 0:
						e_gain.append(e_gain[-1] + elev_dif)
					else:
						e_gain.append(e_gain[-1])

				if ElevationLoss:
					if elev_dif < 0:
						e_loss.append(e_loss[-1] + abs(elev_dif))
					else:
						e_loss.append(e_loss[-1])

				if Time:
					if i > 1:
						time_dif = (pEnd.Time - pStart.Time).total_seconds()
						time_measure.append(time_measure[-1] + time_dif)
					else:
						time_measure.append(0)

				if Speed:
					if i > 1:
						time_dif = (pEnd.Time - pStart.Time).total_seconds()
						speed = calculated_distance / time_dif
						speed_measure.append(speed)
					else:
						speed_measure.append(0)

				if Slope:
					if calculated_distance != 0:
						delta = elev_dif / calculated_distance 
					else:
						delta = 0
					slope_measure.append(delta)

				dist.append(dist[-1] + calculated_distance)

		if Distance:
			if columnLabels != None and 'dist' in columnLabels:
				dataframe.loc[:, columnLabels['dist']] = dist
			elif formula == "haversine":
				if includeElevation:
					dataframe.loc[:, "3dHavDistance"] = dist
				else:
					dataframe.loc[:, "2dHavDistance"] = dist
			else:
				if includeElevation:
					dataframe.loc[:, "3dVinDistance"] = dist
				else:
					dataframe.loc[:, "2dVinDistance"] = dist

		if ElevationGain:
			if columnLabels != None and 'e_gain' in columnLabels:
				dataframe.loc[:, columnLabels['e_gain']] = e_gain
			else:
				dataframe.loc[:, "ElevationGain"] = e_gain

		if ElevationLoss:
			if columnLabels != None and 'e_loss' in columnLabels:
				dataframe.loc[:, columnLabels['e_loss']] = e_loss
			else:
				dataframe.loc[:, "ElevationLoss"] = e_loss

		if Time:
			if columnLabels != None and 'time' in columnLabels:
				dataframe.loc[:, columnLabels['time']] = time_measure
			else:
				dataframe.loc[:, "Time"] = time_measure

		if Speed:
			if columnLabels != None and 'speed' in columnLabels:
				dataframe.loc[:, columnLabels['speed']] = speed_measure
			else:
				dataframe.loc[:, "Speed"] = speed_measure

		if Slope:
			if columnLabels != None and 'slope' in columnLabels:
				dataframe.loc[:, columnLabels['slope']] = slope_measure
			else:
				dataframe.loc[:, 'Slope'] = slope_measure

		
		return dataframe


	def findTotalDistance(self, dataframe, formula = "vincenty", includeElevation = True, columnLabel = None):
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

		if columnLabel != None:
			dataframe[columnLabel] = dist
		elif formula == "haversine":
			if includeElevation:
				dataframe.loc[:, "3dHavDistance"] = dist
			else:
				dataframe.loc[:, "2dHavDistance"] = dist
		else:
			if includeElevation:
				dataframe.loc[:, "3dVinDistance"] = dist
			else:
				dataframe.loc[:, "2dVinDistance"] = dist

		return dist[-1]



	def findTotalElevationGain(self, dataframe, columnLabel = None):
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
		if columnLabel != None:
			dataframe.loc[:, columnLabel] = elev
		else:
			dataframe.loc[:, "elevationGain"] = elev

		return elev[-1]


	def findTotalElevationLoss(self, dataframe, columnLabel = None):
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
		if columnLabel != None:
			dataframe.loc[:, columnLabel] = elev
		else:
			dataframe.loc[:, "elevationLoss"] = elev

		return elev[-1]


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
		preceding_slope = []
		
		i = 0
		while i < len(points)-1:
			turns.append([])
			preceding_slope.append(0)
			sumAngle = 0
			if i < 2:
				pass
				startPoint = 0
			else:
				startPoint = i-2
				newPointStart = i-1
				newPointEnd = i
				referenceAngle = self.findCourse(points[startPoint], points[newPointStart])
				newAngle = self.findCourse(points[newPointStart], points[newPointEnd])
				angle_diff = newAngle - referenceAngle
				sumAngle += angle_diff
				if(angle_diff > threshold):
					while (angle_diff > threshold) and (newPointEnd < len(points)-1):
						try:
							turns[-1].append(points[newPointEnd])
							turns[-1].append(newAngle)
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
								preceding_slope[-1] = startPoint

							

						except IndexError:
							i = len(points)

				elif angle_diff < (-1 * threshold):
					while (angle_diff < (-1 * threshold)) and (newPointEnd < len(points)-1):
						try:
							turns[-1].append(points[newPointEnd])
							turns[-1].append(newAngle)
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
								preceding_slope.append(startPoint)

							

						except IndexError:
							i = len(points)
			if turns[-1] == [] or len(turns[-1]) <= (minLength * 2 + 1):
				del turns[-1]
				del preceding_slope[-1]
				
			i+=1
		return [turns, preceding_slope]

	def findPrecedingSlope(self, turnLocations, dataframe, search_slope_threshold = 5):
		avgSlopes = []
		for turnLoc in turnLocations:
			if turnLoc > search_slope_threshold:
				pointRange = dataframe.iloc[turnLoc-search_slope_threshold:turnLoc+1, -1]
				totalPRSlope = 0
				for point in pointRange:
					totalPRSlope += point
				avgPRSlope = totalPRSlope / len(pointRange)
				avgSlopes.append(avgPRSlope)
		return avgSlopes





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

	def findIntervalsByNum(self, track, intervalCount = 3):
		"""

		"""
		interval_length = len(track) // intervalCount
		intervals = {}
		for i in range(intervalCount):
			if i < (intervalCount-1):
				intervals["interval " + str(i)] = track.iloc[interval_length * i : interval_length * (i+1), :].copy()
			else:
				intervals["interval " + str(i)] = track.iloc[interval_length * i:, :].copy()
		return intervals

	def progress(self, count, total, status=''):
		bar_len = 60
		filled_len = int(round(bar_len * count / float(total)))
		percents = round(100.0 * count / float(total), 1)
		bar = '=' * filled_len + '-' * (bar_len - filled_len)
		sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
		sys.stdout.flush()


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

		print("haversine 2d distance: " + str(test.findTotalDistance(activity[i], includeElevation = False)))
		print("haversine 3d distance: " + str(test.findTotalDistance(activity[i])))
		print("vincenty 2d distance: " + str(test.findTotalDistance(activity[i], formula = "vincenty", includeElevation = False)))
		print("vincenty 3d distance: " + str(test.findTotalDistance(activity[i], formula = "vincenty")))

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
		print("amount of turns: " + str(len(turns[0])))


		s = 0
		degree = 0
		for turn in turns[0]:
			s += len(turn)
			degree += abs(turn[-1])
		avgDegree = degree/len(turns)
		avgLength = s/len(turns[0])
		print("average length of turns: " + str(avgLength))
		print("average degree of turns: " + str(avgDegree))
		print("average distance of points: " + str(dis / len(activity[i])))




		print("FINISHED CALCULATIONS FOR: " + str(i))
		print("\n" + "--------------------------------------" + "\n")


