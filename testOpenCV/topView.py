import cv2
import numpy as np

def findHomoMatrix(cornerPts):
	# if having corners points, then do:
	# src = cornerPts
	src = np.array([
			[130, 550],
			[1630, 575],
			[1760, 860],
			[0, 800]], dtype = "float32")

	dst = np.array([
			[72, 60],
			[470, 60],
			[470, 258],
			[72, 258]], dtype = "float32")

	M = cv2.getPerspectiveTransform(src, dst)
	return M

def topDownView(image, homoMatrix, playerPts):
	# assume that the first 2 players are in one team, and the rest in another
	#image = cv2.imread('test.jpg')
	topViewArt = cv2.imread('court.jpg')

	maxWidth = 470-72
	maxHeight = 258-60

	# the diff between getPerspectiveTransform and findHomography is:
	# findHomography is more rigorous, meaning if the point is not so 'good',
	# it will be discarded
	warped = cv2.warpPerspective(image, homoMatrix, (maxWidth, maxHeight))

	mappedPlayerPos = []
	index = 0
	for pts in playerPts:
		temp = pts.tolist()
		temp.append(1)
		pts = np.asarray(temp)
		newPos = homoMatrix.dot(pts).tolist()
		newPos = [np.int(ele/np.float(newPos[2])) for ele in newPos]
		newPos = newPos[:2]
		if index < 2:
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 10, (0, 0, 255), -1)
		else:
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 10, (255, 0, 0), -1)
		mappedPlayerPos.append(newPos)
		index += 1

	#mappedPlayerPos = np.asarray(mappedPlayerPos)

	return topViewArt, mappedPlayerPos

if (__name__ == '__main__'):
	image = cv2.imread('test.jpg')
	cornerPts = []
	homoMatrix = findHomoMatrix(cornerPts)
	playerPts = np.array([
		[342, 777],
		[417, 861],
		[912, 708],
		[1134, 717]], dtype="float32")
	topViewArtNew, mappedPlayerPos = topDownView(image, homoMatrix, playerPts)
	cv2.imwrite('testNew.jpg', topViewArtNew)
	pass