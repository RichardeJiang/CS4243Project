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

def topDownView(image, homoMatrix, playerPts):
	# assume that the first 2 players are in one team, and the rest in another
	topViewArt = cv2.imread('court.jpg')

	# the diff between getPerspectiveTransform and findHomography is:
	# findHomography is more rigorous, meaning if the point is not so 'good',
	# it will be discarded

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
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (255, 255, 255), -1)
		else:
			# cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (255, 255, 255), -1)
			cv2.circle(topViewArt, (newPos[0], newPos[1]), 8, (255, 255, 255), -1)
		mappedPlayerPos.append(newPos)
		index += 1

	return topViewArt, mappedPlayerPos

def on_mouse(event,x,y,flag,params):
	global ready
	global saved
	global halted
	global initpos
	if event == cv2.EVENT_RBUTTONDOWN:
		print count,x,y
		touchlist.append([count,x,y])
		print 'touched!'
	if event == cv2.EVENT_LBUTTONDOWN:
		if not ready:
			print x,y
			initpos = [x,y]
			print 'clicked!'
			ready = True
		else:
			print count,x,y
			jumplist.append([count,x,y])
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

	testTopViewList = []
	playerPosList = [[[0, 0],[0, 0],[0, 0],[0, 0]]]
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

		frameCount = np.int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

		listActualSize = frameCount

		grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#prepare the matrix
		cornerPts = np.float32(np.array([[[196, 62]],[[440, 137]],[[204, 290]],[[48, 88]]]))
		featurePts = np.float32(np.array([[[36.,135.]],[[294.,80.]],[[122.,185.]],[[224.,70.]],[[172.,206.]],[[134.,158.]],[[60.,142.]],[[80.,170.]],[[95.,218.]],[[27.,218.]],[[45.,186.]],[[28.,264.]],
			[[354., 69.]],[[230.,43.]]]))	
		
		homoMatrix = findHomoMatrixTopDown(cornerPts)

		curr = cornerPts

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
			k = cv2.waitKey(50) & 0xff
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
			playerPosList = playerPosList * listActualSize

		for mouseListIndex in range(0, listActualSize):
			playerPosList[mouseListIndex][iterateIndex] = mouselist[mouseListIndex]

		jumplistTotal.append(list(jumplist))
		touchlistTotal.append(list(touchlist))

		cap.release()

	#playerPosList = smoothPlayerPosData(playerPosList)

	frameCount = frameCountCopy

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