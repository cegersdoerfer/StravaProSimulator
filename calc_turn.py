import math

def findTurns(points, threshold = 10):
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
			referenceAngle = findCourse(points[startPoint], points[newPointStart])
			newAngle = findCourse(points[newPointStart], points[newPointEnd])
			angle_diff = newAngle - referenceAngle
			sumAngle += angle_diff
			if(angle_diff > threshold):
				while angle_diff > threshold:
					turns[-1].append(points[newPointEnd])
					turns[-1].append(newAngle)
					try:
						newPointStart += 1
						newPointEnd += 1
						newAngle = findCourse(points[newPointStart], points[newPointEnd])
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
						newAngle = findCourse(points[newPointStart], points[newPointEnd])
						angle_diff = newAngle - turns[-1][-1]
						if newAngle > 270 and turns[-1][-1] < 90:
							angle_diff -= 360
						sumAngle += angle_diff
						if angle_diff > (-1*threshold):
							i = newPointEnd
							turns[-1].append(sumAngle)


					except IndexError:
						i = len(points)
		if turns[-1] == [] or len(turns[-1]) <= 3:
			del turns[-1]
		i+=1
	return turns




def findCourse(point1, point2):
	convertFactor = 180 / math.pi	
	x_diff = point2[0] - point1[0]
	y_diff = point2[1] - point1[1]
	angle = math.atan2(y_diff , x_diff)
	#print(angle * convertFactor)
	if angle < 0:
		angle = angle + (2 * math.pi)

	return angle * convertFactor


#print(findTurn([[0,0],[1,.5],[2,2],[3,3.5],[4,5],[5,7],[6,8],[7,10],[6,11],[5,11],[4,11],[3,11],[2,12],[1,12]]))
turns = findTurns([[0,0],[1,-1],[2,-2],[3,-2.5],[4,-2],[5,-1],[6,0],[7,1],[8,1.5],[9,1],[10,0],[11,-1],[12,-1.75],[13,-2],[14,-2.1],[15,-2],[16,-1.8],[17,-1.6],[18,-1.4]])
#turnAngles = []
for i in range(len(turns)):
	print(turns[i])
	turnAngle = turns[i][-1]
	print(turnAngle)
	print("\n")









