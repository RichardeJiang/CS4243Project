import cv2
import numpy as np
import numpy.linalg as la
from math import factorial
from scipy.signal import savgol_filter

topViewArtOriginal = cv2.imread('court.jpg')
dashBoardOriginal = cv2.imread('dashBoard.jpg')
distDisplayList = [(240, 214), (240, 408), (652, 214), (652, 408)]
jumpDisplayList = [(240, 280), (240, 474), (652, 280), (652, 474)]
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
			distDisplayList[i], fontFace, 1, color)
		cv2.putText(dashBoard, str("{0:.1f}".format(cumulatedJumps[i])), 
			jumpDisplayList[i], fontFace, 1, color)

	return dashBoard

def smoothList(list,strippedXs=False,degree=20):  

	if strippedXs==True: return Xs[0:-(len(list)-(len(list)-degree+1))]  

	smoothed=[0]*(len(list)-degree+1)  

	for i in range(len(smoothed)):  

		smoothed[i]=sum(list[i:i+degree])/float(degree)  

	return smoothed

def smoothPlayerPosData(data):
	for i in range(4):
		playerX = smoothList([ele[0] for ele in data[i]])
		playerY = smoothList([ele[1] for ele in data[i]])

		playerX = [playerX[0]] * (len(data[i])-len(playerX)) + playerX
		playerY = [playerY[0]] * (len(data[i])-len(playerY)) + playerY

		for j in range(0, len(data[i])):
			data[i][j][0] = playerX[i]
			data[i][j][1] = playerY[i]

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

def topDownView(mappedPlayerPosList, frameIndex):
	# assume that the first 2 players are in one team, and the rest in another
	#topViewArt = cv2.imread('court.jpg')

	# the diff between getPerspectiveTransform and findHomography is:
	# findHomography is more rigorous, meaning if the point is not so 'good',
	# it will be discarded
	topViewArt = topViewArtOriginal.copy()

	count = 0
	playerPts = [] 
	for i in range(4):
		playerPts.append(mappedPlayerPosList[i][frameIndex])
	for pts in playerPts:
		if count < 2:
			cv2.circle(topViewArt, (int(pts[0]), int(pts[1])), 8, (255, 0, 0), -1)
		else:
			cv2.circle(topViewArt, (int(pts[0]), int(pts[1])), 8, (0, 0, 255), -1)
		count += 1

	return topViewArt

def limitRegion(pos, player):
	if player == 0 or player == 1:
		if pos[0] > 270:
			pos[0] = 270
	else:
		if pos[0] < 270:
			pos[0] = 270
	return pos
		
def mapRawPtsToAnime(playerPosList, touchlistTotal, jumplistTotal, homoMatrixList):
	count = len(homoMatrixList)
	mappedPlayerPosList = list(playerPosList)
	mappedTouchList = list(touchlistTotal)
	mappedJumpList = list(jumplistTotal)
	lastactive = []
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

	return mappedPlayerPosList, mappedTouchList, mappedJumpList

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

	frameCountCopy = 0

	for iterateIndex in range(0, 4):

		#playerPos = np.float32(np.array([[[98, 63]],[[174.5, 60]], [[208, 99.5]],[[487.5, 207.5]]]))

		cap = cv2.VideoCapture('beachVolleyball1.mov')
		_, frame = cap.read()

		frameCount = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

		index = 1

		# mouse event vars
		mouselist = []
		jumplist = []
		touchlist = []
		saved = False
		ready = False
		halted = False
		lastsave = [-1,-1]
		initpos = []

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
			if iterateIndex == 0:
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

		playerPosList.append(list(mouselist))

		jumplistTotal.append(list(jumplist))
		touchlistTotal.append(list(touchlist))

		cap.release()

	######has problem
	#playerPosList = smoothPlayerPosData(playerPosList)

	frameCount = frameCountCopy

	testTopViewList = []
	mappedPlayerPosList, mappedTouchList, mappedJumpList = mapRawPtsToAnime(playerPosList, touchlistTotal, jumplistTotal, homoMatrixList)

	boardViewList = []
	cumulatedDistances = [0.,0.,0.,0.]
	cumulatedJumps = [0,0,0,0]

	for frameIndex in range(0, frameCount):
		topViewArtNew = topDownView(mappedPlayerPosList, frameIndex)
		testTopViewList.append(topViewArtNew)
		boardViewNew = generateBoardFrame(mappedPlayerPosList, mappedJumpList, frameIndex, cumulatedDistances, cumulatedJumps)
		boardViewList.append(boardViewNew)

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
	tdvideo = cv2.VideoWriter('topdown.mov',fourcc,fps=59,frameSize=(topViewArtNew.shape[1],topViewArtNew.shape[0]),isColor=1)
	stvideo = cv2.VideoWriter('stats.mov',fourcc,fps=59,frameSize=(boardViewNew.shape[1],boardViewNew.shape[0]),isColor=1)

	for frame in testTopViewList:
		tdvideo.write(frame)
	for frame in boardViewList:
		stvideo.write(frame)

	cv2.destroyAllWindows()
	cap.release()
	pass
