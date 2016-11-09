import cv2
import cv2.cv as cv
import numpy as np
from math import factorial
from scipy.signal import savgol_filter
import numpy.linalg as la

topViewArtOriginal = cv2.imread('court.jpg')
dashBoardOriginal = cv2.imread('dashBoard.jpg')
distDisplayList = [(480, 428), (480, 816), (1304, 428), (1304, 816)]
jumpDisplayList = [(420, 559), (420, 961), (1244, 559), (1244, 961)]
fontFace = cv2.FONT_HERSHEY_SIMPLEX

def calculatePlayerDistancePerFrame(oldFramePos, newFramePos):
	normalValue = 8.0 / 200
	distance = normalValue * la.norm(np.asarray(oldFramePos) - np.asarray(newFramePos))
	return distance

def generateBoardFrame(mappedPlayerPosList, jumpList, frameIndex, cumulatedDistances, cumulatedJumps): 
	dashBoard = dashBoardOriginal.copy()
	for i in range(4):
		if len(jumpList[i]) > 0 and jumpList[i][0][0] == frameIndex:
			cumulatedJumps[i] += 1
			jumpList[i].pop(0)
		if frameIndex == 0:
			cumulatedDistances[i] += calculatePlayerDistancePerFrame(mappedPlayerPosList[i][0], mappedPlayerPosList[i][0])
		else:
			cumulatedDistances[i] += calculatePlayerDistancePerFrame(mappedPlayerPosList[i][frameIndex-1], mappedPlayerPosList[i][frameIndex]) 	
	
	for i in range(0, 4):
		color = (0,0,255)
		if i > 1: 
			color = (255, 0, 0)
		cv2.putText(dashBoard, str("{0:.1f}".format(cumulatedDistances[i])) + ' m', 
			distDisplayList[i], fontFace, 2.5, color, 4)
		cv2.putText(dashBoard, str(int(cumulatedJumps[i])), 
			jumpDisplayList[i], fontFace, 2.5, color, 4)

	return dashBoard

def limitRegion(pos, player):
	if player == 0 or player == 1:
		if int(pos[0]) > 270:
			pos[0] = 270
	else:
		if int(pos[0]) < 270:
			pos[0] = 270
	return pos

def smoothList(list,strippedXs=False,degree=20):  

	if strippedXs==True: return Xs[0:-(len(list)-(len(list)-degree+1))]  

	smoothed=[0]*(len(list)-degree+1)  

	for i in range(len(smoothed)):  

		smoothed[i]=sum(list[i:i+degree])/float(degree)  

	return smoothed

def smoothPlayerPosData(data):
	player1X = smoothList([ele[0][0][0] for ele in data])
	player1Y = smoothList([ele[0][0][1] for ele in data])
	player2X = smoothList([ele[1][0][0] for ele in data])
	player2Y = smoothList([ele[1][0][1] for ele in data])
	player3X = smoothList([ele[2][0][0] for ele in data])
	player3Y = smoothList([ele[2][0][1] for ele in data])
	player4X = smoothList([ele[3][0][0] for ele in data])
	player4Y = smoothList([ele[3][0][1] for ele in data])

	player1X = [player1X[0]] * (len(data) - len(player1X)) + player1X
	player1Y = [player1Y[0]] * (len(data) - len(player1Y)) + player1Y
	player2X = [player2X[0]] * (len(data) - len(player2X)) + player2X
	player2Y = [player2Y[0]] * (len(data) - len(player2Y)) + player2Y
	player3X = [player3X[0]] * (len(data) - len(player3X)) + player3X
	player3Y = [player3Y[0]] * (len(data) - len(player3Y)) + player3Y
	player4X = [player4X[0]] * (len(data) - len(player4X)) + player4X
	player4Y = [player4Y[0]] * (len(data) - len(player4Y)) + player4Y

	for index in range(0, len(data)):
		data[index][0][0][0] = player1X[index]
		data[index][0][0][1] = player1Y[index]
		data[index][1][0][0] = player2X[index]
		data[index][1][0][1] = player2Y[index]
		data[index][2][0][0] = player3X[index]
		data[index][2][0][1] = player3Y[index]
		data[index][3][0][0] = player4X[index]
		data[index][3][0][1] = player4Y[index]

	return data

def findHomoMatrixTopDown(cornerPts):
	# if having corners points, then do:
	# src = cornerPts
	src = cornerPts

	# original corner pts
	dst = np.array([
			[70, 60],
			[470, 60],
			[470, 260],
			[70, 260]], dtype = "float32")

	M = cv2.getPerspectiveTransform(src, dst)
	return M

def findHomographyMatrix(src_pts, dst_pts):
	# ptFrame1 is 3D numpy array, each element is 2D array of the coordinate
	if (len(src_pts) < 4 or len(dst_pts) < 4):
		return
	else:
		homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		return homographyMatrix

def topDownView(mappedPlayerPosList, mappedBallPosList, frameIndex):
	# assume that the first 2 players are in one team, and the rest in another
	topViewArt = topViewArtOriginal.copy()

	# the diff between getPerspectiveTransform and findHomography is:
	# findHomography is more rigorous, meaning if the point is not so 'good',
	# it will be discarded

	for index in range(0, 4):
		x = int(mappedPlayerPosList[index][frameIndex][0])
		y = int(mappedPlayerPosList[index][frameIndex][1])
		
		if index < 2:
			cv2.circle(topViewArt, (x, y), 8, (255, 0, 0), -1)
		else:
			cv2.circle(topViewArt, (x, y), 8, (0, 0, 255), -1)

	ballX = int(mappedBallPosList[frameIndex][0])
	ballY = int(mappedBallPosList[frameIndex][1])

	cv2.circle(topViewArt, (ballX, ballY), 4, (0, 0, 0), -1)

	return topViewArt

def computeBallPosList(mappedPlayerPosList, mappedTouchList, mappedLastBallPtsList, frameCount):
	playerHitList = []
	for playerIndex in range(0, 4):
		for ele in mappedTouchList[playerIndex]:
			ele.insert(1, playerIndex)
			playerHitList.append(list(ele))

	playerHitList.sort(key=lambda x: x[0])

	mappedBallPosList = [[0, 0]] * frameCount
	for playerHit in playerHitList:
		mappedBallPosList[playerHit[0]] = list(playerHit[2:])
	for index in range(0, len(playerHitList)):
		if index == 0:
			playerIndex = playerHitList[0][1]
			for frame in range(0, playerHitList[0][0] - 1):
				#mappedBallPosList[frame] = list(playerHitList[0][2:])
				mappedBallPosList[frame] = list(mappedPlayerPosList[playerIndex][frame])
		else:
			xOffset = playerHitList[index][2] - playerHitList[index-1][2]
			yOffset = playerHitList[index][3] - playerHitList[index-1][3]
			numOfFramesBetween = playerHitList[index][0] - playerHitList[index-1][0]
			xChangeUnit = xOffset / float(numOfFramesBetween)
			yChangeUnit = yOffset / float(numOfFramesBetween)
			for frame in range(playerHitList[index-1][0]+1, playerHitList[index][0]):
				newX = playerHitList[index-1][2] + xChangeUnit * (frame - playerHitList[index-1][0])
				newY = playerHitList[index-1][3] + yChangeUnit * (frame - playerHitList[index-1][0])
				mappedBallPosList[frame] = [newX, newY]

	beforeLastFrame = playerHitList[-1][0]
	xOffsetLastBall = mappedLastBallPtsList[0][0][1] - playerHitList[-1][2]
	yOffsetLastBall = mappedLastBallPtsList[0][0][2] - playerHitList[-1][3]
	lastFrame = mappedLastBallPtsList[0][0][0]
	numOfFrames = lastFrame - beforeLastFrame
	xChangeUnitLast = xOffsetLastBall / float(numOfFrames)
	yChangeUnitLast = yOffsetLastBall / float(numOfFrames)

	for frame in range(playerHitList[-1][0] + 1, len(mappedBallPosList)):
		newXLast = playerHitList[-1][2] + xChangeUnitLast * (frame - playerHitList[-1][0])
		newYLast = playerHitList[-1][3] + yChangeUnitLast * (frame - playerHitList[-1][0])
		mappedBallPosList[frame] = [newXLast, newYLast]

	return mappedBallPosList

def mapPts(originPts, homoMatrix):
	originPts.append(1)
	pts = np.asarray(originPts)
	newPos = homoMatrix.dot(pts).tolist()
	newPos = [np.int(ele/np.float(newPos[2])) for ele in newPos]
	newPos = newPos[:2]
	return newPos

def mapRawPtsToAnime(playerPosList, touchlistTotal, jumplistTotal, lastBallPtsList, homoMatrixList):
	count = min(len(homoMatrixList), len(playerPosList[0]))
	mappedPlayerPosList = list(playerPosList)
	mappedTouchList = list(touchlistTotal)
	mappedJumpList = list(jumplistTotal)
	mappedLastBallPtsList = list(lastBallPtsList)
	lastactive = []

	for lastBallPts in lastBallPtsList:
		frameIndex = lastBallPts[0][0]
		homoMatrix = homoMatrixList[frameIndex]
		temp2 = lastBallPts[0][1:]
		mappedLastBallPtsList[lastBallPtsList.index(lastBallPts)][0][1:] = mapPts(list(temp2), homoMatrix)

	for playerIndex in range(4):
		for i in range(len(mappedTouchList[playerIndex])):
			homoMatrix = homoMatrixList[touchlistTotal[playerIndex][i][0]]
			temp = touchlistTotal[playerIndex][i][1:]
			mappedTouchList[playerIndex][i][1:] = cv2.perspectiveTransform(np.float32(np.asarray([[temp]])), homoMatrix).tolist()[0][0]

		for i in range(len(mappedJumpList[playerIndex])):
			homoMatrix = homoMatrixList[jumplistTotal[playerIndex][i][0]]
			temp = jumplistTotal[playerIndex][i][1:]
			mappedJumpList[playerIndex][i][1:] = cv2.perspectiveTransform(np.float32(np.asarray([[temp]])), homoMatrix).tolist()[0][0]

		for frameIndex in range(count):
			homoMatrix = homoMatrixList[frameIndex]
			temp = playerPosList[playerIndex][frameIndex]
			if temp[0] == -1: #out of view
				if lastactive == []:
					lastactive = playerPosList[playerIndex][frameIndex-1]
				temp = lastactive
			else:
				if lastactive != []:
					lastactive = []
			mappedPlayerPosList[playerIndex][frameIndex] = limitRegion(cv2.perspectiveTransform(np.float32(np.asarray([[temp]])), homoMatrix).tolist()[0][0], playerIndex)
			mappedPlayerPosList[playerIndex][frameIndex]

	# original implementation

	# for playerIndex in range(0, 4):

	# 	playerHitFrameList = []
	# 	for playerHit in touchlistTotal[playerIndex]: # len(touchlistTotal[playerIndex])
	# 		playerHitFrameList.append(playerHit[0])

	# 	for frameIndex in range(0, count):
	# 		homoMatrix = homoMatrixList[frameIndex]

	# 		if frameIndex in playerHitFrameList:
	# 			temp1 = touchlistTotal[playerIndex][playerHitFrameList.index(frameIndex)][1:]
	# 			mappedTouchList[playerIndex][playerHitFrameList.index(frameIndex)][1:] = mapPts(list(temp1), homoMatrix)

	# 		temp = playerPosList[playerIndex][frameIndex]
	# 		print 'now temp is: ', temp

	# 		if temp == [-1, -1]:
	# 			for i in range(1, frameIndex):
	# 				temp = playerPosList[playerIndex][frameIndex - i]
	# 				if (temp != [-1, -1]):
	# 					break
	# 		print 'temp is: ', temp
	# 		mappedPlayerPosList[playerIndex][frameIndex] = mapPts(list(temp), homoMatrix)


	return mappedPlayerPosList, mappedTouchList, mappedJumpList, mappedLastBallPtsList

def on_mouse(event,x,y,flag,params):
	global ready
	global saved
	global halted
	global initpos
	if event == cv2.EVENT_RBUTTONDOWN:
		print index,x,y
		touchlist.append([index,x,y])
		print 'touched!'
	if event == cv2.EVENT_LBUTTONDOWN:
		if not ready:
			print x,y
			initpos = [x,y]
			print 'clicked!'
			ready = True
		else:
			print index,x,y
			jumplist.append([index,x,y])
			print 'jumped!'
	if event == cv2.EVENT_MBUTTONDOWN:
		#out of screen
		halted = not halted
		print 'halted: ' + str(halted)
	if flag == (cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_LBUTTONDOWN):
		lastBallPts.append([index,x,y])
	#if event == cv2.EVENT_MOUSEMOVE:
	if ready and not saved:
		saved = True
		if halted:
			mouselist.append([-1,-1])
			print -1,-1
		else:
			mouselist.append([x,y])	
			print x,y	

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

	topViewArt = cv2.imread('court.jpg')

	playerPosList = []
	homoMatrixList = []
	frameList = []

	mouselist = []
	jumplistTotal = []
	touchlistTotal = []
	lastBallPtsList = []

	frameCountCopy = 0

	for iterateIndex in range(0, 5):

		#playerPos = np.float32(np.array([[[98, 63]],[[174.5, 60]], [[208, 99.5]],[[487.5, 207.5]]]))

		cap = cv2.VideoCapture('beachVolleyball1.mov')
		_, frame = cap.read()

		frameCount = np.int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

		listActualSize = frameCount

		grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#prepare the matrix
		cornerPts = np.float32(np.array([[[196, 62]],[[440, 137]],[[204, 290]],[[48, 88]]]))
		featurePts = np.float32(np.array([[[36.,135.]],[[294.,80.]],[[122.,185.]],[[224.,70.]],[[172.,206.]],[[134.,158.]],[[60.,142.]],[[80.,170.]],[[95.,218.]],[[27.,218.]],[[45.,186.]],[[28.,264.]],
			[[354., 69.]],[[230.,43.]]]))	
		
		homoMatrix = findHomoMatrixTopDown(cornerPts)

		curr = cornerPts

		if iterateIndex == 0:
			homoMatrixList.append(homoMatrix.copy())
		frameList.append(grayOld.copy())

		index = 0

		# mouse event vars
		mouselist = []
		jumplist = []
		touchlist = []
		saved = False
		ready = False
		halted = False
		lastsave = [-1,-1]
		initpos = []
		lastBallPts = []

		cv2.namedWindow('frame')
		cv2.setMouseCallback('frame', on_mouse)

		while(_):
			_, frame = cap.read()
			if not _:
				for i in range(index-len(mouselist)):
					mouselist = [initpos] + mouselist
				break

			#cv2.namedWindow('frame')
			saved = False
			#cv2.setMouseCallback('frame', on_mouse)
			if len(mouselist) > 0 and mouselist[-1] == lastsave:
				mouselist.append(lastsave)
				print mouselist[-1]
			if len(mouselist) > 0:
				lastsave = mouselist[-1]
			cv2.imshow('frame',frame)
			k = cv2.waitKey(10) & 0xff
			if k == 27:
				break
			# Now update the previous frame and previous points

			grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			featureNewPts, st2, err2 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, featurePts, None, **lk_params)

			good_new = featureNewPts[st2==1]
			good_old = featurePts[st2==1]
			
			hi = findHomographyMatrix(good_old,good_new)
			curr = cv2.perspectiveTransform(curr, hi)
			homoMatrix = findHomoMatrixTopDown(curr)

			homoMatrixList.append(homoMatrix.copy())
			frameList.append(grayOld)
			grayOld = grayNew.copy()
			featurePts = good_new.reshape(-1,1,2)
			index += 1

		if iterateIndex == 0:
			listActualSize = len(mouselist)
			frameCountCopy = listActualSize
		# 	playerPosList = playerPosList * listActualSize

		# for mouseListIndex in range(0, listActualSize):
		# 	playerPosList[mouseListIndex][iterateIndex] = mouselist[mouseListIndex]

		if iterateIndex != 4:
			playerPosList.append(list(mouselist))

			jumplistTotal.append(list(jumplist))
			touchlistTotal.append(list(touchlist))

		else:
			lastBallPtsList.append(list(lastBallPts))

		cap.release()

	#playerPosList = smoothPlayerPosData(playerPosList)

	frameCount = frameCountCopy

	print np.asarray(playerPosList)

	testTopViewList = []
	mappedPlayerPosList, mappedTouchList, mappedJumpList, mappedLastBallPtsList = mapRawPtsToAnime(playerPosList, 
		touchlistTotal, jumplistTotal, lastBallPtsList, homoMatrixList)

	mappedBallPosList = computeBallPosList(mappedPlayerPosList, mappedTouchList, mappedLastBallPtsList, frameCount)

	while(len(mappedBallPosList) < len(mappedPlayerPosList)):
		mappedBallPosList.append(mappedBallPosList[-1])

	frameCount = min(len(mappedPlayerPosList[0]), len(mappedBallPosList))
	boardViewList = []
	cumulatedDistances = [0.,0.,0.,0.]
	cumulatedJumps = [0,0,0,0]

	for frameIndex in range(0, frameCount):
		topViewArtNew = topDownView(mappedPlayerPosList, mappedBallPosList, frameIndex)
		testTopViewList.append(topViewArtNew)
		boardViewNew = generateBoardFrame(mappedPlayerPosList, mappedJumpList, frameIndex, cumulatedDistances, cumulatedJumps)
		boardViewList.append(boardViewNew)

	heightTopdown,widthTopdown = topViewArtNew.shape[:2]
	heightBoard, widthBoard = boardViewNew.shape[:2]
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	tdvideo = cv2.VideoWriter('topdown.mov',fourcc,fps=59,frameSize=(widthTopdown,heightTopdown),isColor=1)
	stvideo = cv2.VideoWriter('stats.mov',fourcc,fps=59,frameSize=(widthBoard,heightBoard),isColor=1)

	for frame in testTopViewList:
		tdvideo.write(frame)

	for frame in boardViewList:
		stvideo.write(frame)

	cv2.destroyAllWindows()
	cap.release()
	pass