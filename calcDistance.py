import cv2
import cv2.cv as cv
import numpy as np
import numpy.linalg as la

def checkJump(firstPlayerPosList, secondPlayerPosList, playerIndex):
	# the initial idea to check jumping is to see every 20 frames
	# if the offset in y direction is above a certain threshold and in x direction is below a threshold, 
	# consider it as a jump
	playerBeforePosX = firstPlayerPosList[playerIndex][0][0]
	playerAfterPosX = secondPlayerPosList[playerIndex][0][0]
	playerBeforePosY = firstPlayerPosList[playerIndex][0][1]
	playerAfterPosY = secondPlayerPosList[playerIndex][0][1]

	if playerIndex < 2:
		if playerBeforePosY - playerAfterPosY >= 14 and abs(playerBeforePosX - playerAfterPosX) <= 10:
			return True
	else:
		if playerBeforePosY - playerAfterPosY >= 25 and abs(playerBeforePosX - playerAfterPosX) <= 10:
		#if abs(playerbeforePosY - playerAfterPosY) >= 20:
			return True

	return False

def calculatePlayerDistancePerFrame(oldFramePos, newFramePos):
	distances = [0.0,0.0,0.0,0.0]
	normalValue = 8.0 / 200
	for index in range(0, 4):
		distance = normalValue * la.norm(oldFramePos[index][0] - newFramePos[index][0])
		distances[index] = distance
	return distances

if (__name__ == "__main__"):

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

	#need to be filled in
	playerPos = np.float32(np.array([[[91, 209]],[[152, 133]], [[309, 171]],[[538, 161]]]))
	playerPos = np.float32(np.array([[[91, 208]],[[152, 132]], [[309, 171]],[[536, 159]]]))

	jumpTrackingPos = np.float32(np.array([[[98, 63]],[[174.5, 60]], [[209, 80.5]],[[487.5, 207.5]]]))

	cap = cv2.VideoCapture('panorama.mov')
	_, frame = cap.read()

	capJump = cv2.VideoCapture('beachVolleyball1.mov')
	ret, frameJump = capJump.read()

	grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	grayOldJump = cv2.cvtColor(frameJump, cv2.COLOR_BGR2GRAY)

	cumulatedDistances = [0.0, 0.0, 0.0, 0.0]

	frameIndex = 0
	jumpFlag = [False] * 4
	afterJumpCounter = [0] * 4
	checkJumpWindow = 28

	jumpTrackingPosList = []
	jumpTrackingPosList.append(jumpTrackingPos.copy())


	cumulatedJumps = [0, 0, 0, 0]

	dashBoard = cv2.imread('dashBoard.jpg')
	dashBoardOriginal = dashBoard.copy()
	height, width = dashBoard.shape[:2]
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('stats1.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
	orgList = [(480, 428), (480, 816), (1304, 428), (1304, 816)]
	orgListJump = [(420, 559), (420, 961), (1244, 559), (1244, 961)]
	fontFace = cv2.FONT_HERSHEY_SIMPLEX
	testCalcDistanceList = []

	while(_ and ret):
		_, frame = cap.read()
		ret, frameJump = capJump.read()

		if not _ or not ret:
			break

		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grayNewJump = cv2.cvtColor(frameJump, cv2.COLOR_BGR2GRAY)

		# calculate new player position coordinates
		playerNewPos, st, err = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, playerPos, None, **lk_params)
		jumpTrackingNewPos, st1, err1 = cv2.calcOpticalFlowPyrLK(grayOldJump, grayNewJump, jumpTrackingPos, None, **lk_params)

		goodOld = playerPos[st==1]
		goodNew = playerNewPos[st==1]

		goodNewJump = jumpTrackingNewPos[st1==1]

		distances = calculatePlayerDistancePerFrame(goodOld, goodNew)

		for playerIndex in range(0, 4):
			if frameIndex >= checkJumpWindow and jumpFlag[playerIndex] == False and afterJumpCounter[playerIndex] >= checkJumpWindow:
				jumpFlag[playerIndex] = checkJump(jumpTrackingPosList[frameIndex - checkJumpWindow], jumpTrackingPosList[frameIndex], playerIndex)
				if jumpFlag[playerIndex] == True:
					print 'player ' + str(playerIndex + 1) + ' jumps!!!'
					cumulatedJumps[playerIndex] += 1
					afterJumpCounter[playerIndex] = 0
					jumpFlag[playerIndex] = False

			if playerIndex < 2:
				cv2.putText(dashBoard, str(int(cumulatedJumps[playerIndex])), 
					orgListJump[playerIndex], fontFace, 2.5, (0, 0, 255), 4)
			else:
				cv2.putText(dashBoard, str(int(cumulatedJumps[playerIndex])), 
					orgListJump[playerIndex], fontFace, 2.5, (255, 0, 0), 4)

		for index in range(0, 4):
			cumulatedDistances[index] += distances[index]

		for index in range(0, 4):

			if index < 2:
				cv2.putText(dashBoard, str("{0:.1f}".format(cumulatedDistances[index])) + ' m', 
					orgList[index], fontFace, 2.5, (0, 0, 255), 4)
			else:
				cv2.putText(dashBoard, str("{0:.1f}".format(cumulatedDistances[index])) + ' m', 
					orgList[index], fontFace, 2.5, (255, 0, 0), 4)

		testCalcDistanceList.append(dashBoard.copy())
		video.write(dashBoard)
		dashBoard = dashBoardOriginal.copy()

		grayOld = grayNew.copy()
		grayOldJump = grayNewJump.copy()
		playerPos = goodNew.reshape(-1, 1, 2)

		jumpTrackingPos = goodNewJump.reshape(-1, 1, 2)

		jumpTrackingPosList.append(jumpTrackingNewPos.copy())
		frameIndex += 1
		afterJumpCounter = [ele + 1 for ele in afterJumpCounter]

	# for board in testCalcDistanceList:
	# 	video.write(board)

	cv2.destroyAllWindows()
	cap.release()
	pass