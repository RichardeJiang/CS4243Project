import cv2
import cv2.cv as cv
import numpy as np
from math import factorial
from scipy.signal import savgol_filter

def smoothList(list,strippedXs=False,degree=20):  

	if strippedXs==True: return Xs[0:-(len(list)-(len(list)-degree+1))]  

	smoothed=[0]*(len(list)-degree+1)  

	for i in range(len(smoothed)):  

		smoothed[i]=sum(list[i:i+degree])/float(degree)  

	return smoothed

def smoothListGaussian(list,strippedXs=False,degree=5):  

	window=degree*2-1  
	weight=np.array([1.0]*window)  
	weightGauss=[]  

	for i in range(window):  

		i=i-degree+1  

		frac=i/float(window)  

		gauss=1/(np.exp((4*(frac))**2))  

		weightGauss.append(gauss)  

	weight=np.array(weightGauss)*weight  

	smoothed=[0.0]*(len(list)-window)  

	for i in range(len(smoothed)):  

		smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)  

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

	# player1X = savgol_filter(np.asarray([ele[0][0][0] for ele in data]),21,5)
	# player1Y = savgol_filter(np.asarray([ele[0][0][1] for ele in data]),21,5)
	# player2X = savgol_filter(np.asarray([ele[1][0][0] for ele in data]),21,5)
	# player2Y = savgol_filter(np.asarray([ele[1][0][1] for ele in data]),21,5)
	# player3X = savgol_filter(np.asarray([ele[2][0][0] for ele in data]),21,5)
	# player3Y = savgol_filter(np.asarray([ele[2][0][1] for ele in data]),21,5)
	# player4X = savgol_filter(np.asarray([ele[3][0][0] for ele in data]),21,5)
	# player4Y = savgol_filter(np.asarray([ele[3][0][1] for ele in data]),21,5)

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

# def smoothPlayerPosData(data):
# 	return savgol_filter(data)

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
		if index < 3:
			temp[1] += 20
		else:
			temp[1] += 45
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

def checkJump(firstPlayerPosList, secondPlayerPosList, playerIndex):
	# the initial idea to check jumping is to see every 20 frames
	# if the offset in y direction is above a certain threshold and in x direction is below a threshold, 
	# consider it as a jump
	playerBeforePosX = firstPlayerPosList[playerIndex][0][0]
	playerAfterPosX = secondPlayerPosList[playerIndex][0][0]
	playerBeforePosY = firstPlayerPosList[playerIndex][0][1]
	playerAfterPosY = secondPlayerPosList[playerIndex][0][1]

	if playerIndex < 3:
		if playerBeforePosY - playerAfterPosY >= 10 and abs(playerBeforePosX - playerAfterPosX) <= 10:
			return True
	else:
		if playerBeforePosY - playerAfterPosY >= 25 and abs(playerBeforePosX - playerAfterPosX) <= 10:
		#if abs(playerbeforePosY - playerAfterPosY) >= 20:
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
	playerPos = np.float32(np.array([[[98, 63]],[[174.5, 60]], [[208, 99.5]],[[487.5, 207.5]]]))

	testTopViewList = []

	# use the shirt to check the jumps

	cap = cv2.VideoCapture('beachVolleyball1.mov')
	_, frame = cap.read()

	frameCount = np.int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

	grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#prepare the matrix
	cornerPts = np.float32(np.array([[[196, 62]],[[440, 137]],[[204, 290]],[[48, 88]]]))
	featurePts = np.float32(np.array([[[36.,135.]],[[294.,80.]],[[122.,185.]],[[224.,70.]],[[172.,206.]],[[134.,158.]],[[60.,142.]],[[80.,170.]],[[95.,218.]],[[27.,218.]],[[45.,186.]],[[28.,264.]],
[[354., 69.]],[[230.,43.]]]))	
	
	homoMatrix = findHomoMatrixTopDown(cornerPts)
	topViewArtNew, mappedPlayerPos = topDownView(grayOld, homoMatrix, playerPos)

	playerPosList = []
	playerPosList.append(playerPos.copy())

	testTopViewList.append(topViewArtNew.copy())
	index = 0
	jumpFlag = [False] * 4
	afterJumpCounter = [0] * 4
	checkJumpWindow = 29
	updateWindow = 40
	updateFutureFlag = [False] * 4

	curr = cornerPts

	homoMatrixList = []
	frameList = []
	homoMatrixList.append(homoMatrix.copy())
	frameList.append(grayOld.copy())

	while(_):
		_, frame = cap.read()
		if not _:
			break

		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		playerNewPos, st1, err1 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, playerPos, None, **lk_params)
		featureNewPts, st2, err2 = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, featurePts, None, **lk_params)

		good_new = featureNewPts[st2==1]
		good_old = featurePts[st2==1]
		
		hi = findHomographyMatrix(good_old,good_new)
		curr = cv2.perspectiveTransform(curr, hi)
		homoMatrix = findHomoMatrixTopDown(curr)

		playerNewPosCopy = playerNewPos.copy()


		for playerIndex in range(0, 4):
			if index >= checkJumpWindow and jumpFlag[playerIndex] == False and afterJumpCounter[playerIndex] >= checkJumpWindow:
				jumpFlag[playerIndex] = checkJump(playerPosList[index - checkJumpWindow], playerPosList[index], playerIndex)
				if jumpFlag[playerIndex] == True:
					print 'player ' + str(playerIndex + 1) + ' jumps!!!'
					playerNewPosCopy[playerIndex] = playerPosList[index - checkJumpWindow][playerIndex]
					afterJumpCounter[playerIndex] = 0
					jumpFlag[playerIndex] = False
					updateFutureFlag[playerIndex] = True

					for newPageIndex in range(0, checkJumpWindow):
						playerPosList[index - newPageIndex][playerIndex] = playerPosList[index - checkJumpWindow][playerIndex]

		if True in updateFutureFlag:
			for playerIndex in range(0, 4):
				if updateFutureFlag[playerIndex] == True:
					if afterJumpCounter[playerIndex] < checkJumpWindow:
						playerNewPosCopy[playerIndex] = playerPosList[-1][playerIndex]
					else:
						updateFutureFlag[playerIndex] = False

		#topViewArtNew, mappedPlayerPos = topDownView(grayNew, homoMatrix, playerNewPos)
		playerPosList.append(playerNewPosCopy.copy())
		homoMatrixList.append(homoMatrix.copy())
		frameList.append(grayOld)
		grayOld = grayNew.copy()
		playerPos = playerNewPos.copy()
		featurePts = good_new.reshape(-1,1,2)
		index += 1
		afterJumpCounter = [ele + 1 for ele in afterJumpCounter]

	playerPosList = smoothPlayerPosData(playerPosList)

	for frameIndex in range(0, frameCount):
		topViewArtNew, mappedPlayerPos = topDownView(frameList[frameIndex], homoMatrixList[frameIndex], playerPosList[frameIndex])
		testTopViewList.append(topViewArtNew)

	height,width = topViewArtNew.shape[:2]
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('panorama.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)

	for warp in testTopViewList:
		video.write(warp)

	cv2.destroyAllWindows()
	cap.release()
	pass