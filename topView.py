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

	src = cornerPts

	dst = np.array([
			[70, 60],
			[470, 60],
			[470, 260],
			[70, 260]], dtype = "float32")

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
		temp = pts[0].tolist()
		temp.append(1)
		pts = np.asarray(temp)
		newPos = homoMatrix.dot(pts).tolist()
		newPos = [np.int(ele/np.float(newPos[2])) for ele in newPos]
		newPos = newPos[:2]
		if index < 2:
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (0, 0, 255), -1)
		else:
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (255, 0, 0), -1)
		mappedPlayerPos.append(newPos)
		index += 1

	#mappedPlayerPos = np.asarray(mappedPlayerPos)

	return topViewArt, mappedPlayerPos

def checkJump(firstPlayerPosList, secondPlayerPosList):
	# the initial idea to check jumping is to see every 20 frames
	# if the offset in y direction is above a certain threshold and in x direction is below a threshold, 
	# consider it as a jump
	for index in range(0, 4):
		playerBeforePosX = firstPlayerPosList[index][0][0]
		playerAfterPosX = secondPlayerPosList[index][0][0]
		playerbeforePosY = firstPlayerPosList[index][0][1]
		playerAfterPosY = secondPlayerPosList[index][0][1]

		if playerbeforePosY - playerAfterPosY >= 30 and
			abs(playerBeforePosX - playerAfterPosX) <= 8:
			return True

	return False

if (__name__ == '__main__'):

	# use a list to store all generated frames first since we need to check whether it's a jump or not
	# and if it is then need to revert the previous 20 frames
		# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
		qualityLevel = 0.3,
		minDistance = 7,
		blockSize = 7 )

	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
		maxLevel = 2,
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	#need to be filled in; use this to check the player position
	playerPos = np.float32(np.array([[[188, 168]],[[327, 146]], [[980, 460]],[[410, 225]]]))

	testTopViewList = []

	# use the shirt to check the jumps
	playerShirtPos = np.float32(np.array())

	cap = cv2.VideoCapture('panoramadiagonal.mov')
	_, frame = cap.read()

	grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	index = 0

	#prepare the matrix
	cornerPts = np.float32(np.array([[393, 126],[876, 279],[405, 579],[95, 177]]))
	homoMatrix = findHomoMatrix(cornerPts)
	topViewArtNew, mappedPlayerPos = topDownView(grayOld, homoMatrix, playerPos)

	testTopViewList.append(topViewArtNew.copy())

	while(_):
		_, frame = cap.read()
		if not _:
			break

		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		playerNewPos, st1, err1 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, playerPos, None, **lk_params)
		#playerNewShirtPos, st2, err2 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, playerShirtPos, None, **lk_params)

		goodPlayerOld = playerPos(st1 == 1)
		goodPlayerNew = playerNewPos(st1 == 1)

		#goodShirtOld = playerShirtPos(st2 == 1)
		#goodShirtNew = playerNewShirtPos(st2 == 1)

		# topdown view to be implemented here
		# also check the crrent frame with the one 20 frames ahead/before for jumps

		topViewArtNew, mappedPlayerPos = topDownView(grayNew, homoMatrix, playerNewPos)
		testTopViewList.append(topViewArtNew.copy())
		grayOld = grayNew.copy()
		playerPos = playerNewPos.copy()
		#playerShirtPos = playerNewShirtPos


	height,width = topViewArtNew.shape[:2]
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('panorama3.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
	for warp in testTopViewList:
		video.write(warp)

	# original implementation
	# image = cv2.imread('test.jpg')
	# cornerPts = []
	# homoMatrix = findHomoMatrix(cornerPts)
	# playerPts = np.array([
	# 	[342, 777],
	# 	[417, 861],
	# 	[912, 708],
	# 	[1134, 717]], dtype="float32")
	# topViewArtNew, mappedPlayerPos = topDownView(image, homoMatrix, playerPts)
	# cv2.imwrite('testNew.jpg', topViewArtNew)
	cv2.destroyAllWindows()
	cap.release()
	pass