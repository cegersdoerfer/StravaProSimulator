import gpxpy
import os
import pandas as pd

class proDataImporter():
	def __init__(Path = None):
		self.Path = Path if not Path is None else "/Desktop/Strava-ProData"

	def getAllActivities(self):
		files = os.listdir(self.Path)

	def getAllGenderedActivities(self, gender):
		pass

	def getSingleAthleteActivites(self, athlete):
		pass

	def getRandomActivitiesByGender(self, gender):
		pass

	def getRandomActivityByAthlete(self, athlete):
		pass

	def parsegpx(self, files = []):
		if len(files) == 0:
			files = self.getAllActivites()

		dataPoints = []
		for file in files
			with open(self.Path, 'r') as gpxFile:
				gpx = gpxpy.parse(gpxFile)
				for track in gpx.tracks:
					for segment in track.segments:
						for point in segment.points:
							dic = {"Time" : point.time,
								   "Latitude" : point.latitude,
								   "Longitude" : point.longitude,
								   "Elevation" : point.Elevation
								   }
							dataPoints.append(dic)