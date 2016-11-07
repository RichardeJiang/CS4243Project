import cv2
import cv2.cv as cv
import numpy as np

def findHomoMatrix(cornerPts, featurePts):
	# if having corners points, then do:
	# src = cornerPts
	src = np.array([
			[130, 550],
			[1630, 575],
			[1760, 860],
			[0, 800]], dtype = "float32")

	src = cornerPts

	# original corner pts
	dst = np.array([
			[70, 60],
			[470, 60],
			[470, 260],
			[70, 260]], dtype = "float32")

	# film 1
	dst = featurePts

	M = cv2.getPerspectiveTransform(src, dst)
	return M

def findFeaturePtsMapping(cornerPts, featurePts):
	src = cornerPts
	dst = np.array([
			[70, 60],
			[470, 60],
			[470, 260],
			[70, 260]], dtype = "float32")
	homoMatrix = cv2.getPerspectiveTransform(src, dst)
	mappedFeaturePts = []
	for pts in featurePts:
		temp = pts[0].tolist()
		temp.append(1)
		pts = np.asarray(temp)
		newPos = homoMatrix.dot(pts).tolist()
		newPos = [np.int(ele/np.float(newPos[2])) for ele in newPos]
		newPos = newPos[:2]
		mappedFeaturePts.append(newPos)
	return np.float32(np.asarray(mappedFeaturePts)).reshape(-1, 1, 2)

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
		if index < 2:
			temp[1] += 20
		else:
			temp[1] += 40
		temp.append(1)
		pts = np.asarray(temp)
		newPos = homoMatrix.dot(pts).tolist()
		newPos = [np.int(ele/np.float(newPos[2])) for ele in newPos]
		newPos = newPos[:2]
		if index < 2:
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (0, 0, 0), -1)
		else:
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (255, 255, 255), -1)
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

		if abs(playerbeforePosY - playerAfterPosY) >= 20 and abs(playerBeforePosX - playerAfterPosX) <= 8:
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
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	#need to be filled in; use this to check the player position
	playerPos = np.float32(np.array([[[101, 66]],[[172, 57]], [[209, 79]],[[506, 197]]]))

	testTopViewList = []

	# use the shirt to check the jumps

	cap = cv2.VideoCapture('beachVolleyball1.mov')
	_, frame = cap.read()

	grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#prepare the matrix
	cornerPts = np.float32(np.array([[[196, 62]],[[440, 137]],[[204, 290]],[[48, 88]]]))
	featurePts = np.float32(np.array([[[293.5, 83]],[[355, 193]],[[172, 208]],[[46.5, 137]]]))
	mappedFeaturePts = findFeaturePtsMapping(cornerPts, featurePts)
	print mappedFeaturePts

	homoMatrix = findHomoMatrix(cornerPts, mappedFeaturePts)
	topViewArtNew, mappedPlayerPos = topDownView(grayOld, homoMatrix, playerPos)

	playerPosList = []
	playerPosList.append(playerPos.copy())

	testTopViewList.append(topViewArtNew.copy())
	index = 0
	jumpFlag = False
	afterJumpCounter = 0
	checkJumpWindowSize = 15

	while(_):
		_, frame = cap.read()
		if not _:
			break

		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		playerNewPos, st1, err1 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, playerPos, None, **lk_params)
		featureNewPts, st2, err2 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, featurePts, None, **lk_params)

		homoMatrix = findHomoMatrix(featureNewPts, mappedFeaturePts)

		goodPlayerOld = playerPos[st1 == 1]
		goodPlayerNew = playerNewPos[st1 == 1]

		if index >= checkJumpWindowSize and jumpFlag == False and afterJumpCounter>= checkJumpWindowSize:
			jumpFlag = checkJump(playerPosList[index - checkJumpWindowSize], playerPosList[index])
			if jumpFlag == True:
				print 'JUMP!!!'
				afterJumpCounter = 0

		topViewArtNew, mappedPlayerPos = topDownView(grayNew, homoMatrix, playerNewPos)
		testTopViewList.append(topViewArtNew.copy())
		playerPosList.append(playerNewPos.copy())
		grayOld = grayNew.copy()
		playerPos = playerNewPos.copy()
		featurePts = featureNewPts.copy()
		index += 1
		afterJumpCounter += 1


	height,width = topViewArtNew.shape[:2]
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('panorama.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
	for warp in testTopViewList:
		video.write(warp)

	cv2.destroyAllWindows()
	cap.release()
	pass