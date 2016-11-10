import cv2
import numpy as np
import numpy.linalg as la
import sys

video = int(sys.argv[1])
topViewArtOriginal = cv2.imread('court.jpg')
dashBoardOriginal = cv2.imread('dashBoard.jpg')
distDisplayList = [(480, 428), (480, 816), (1304, 428), (1304, 816)]
jumpDisplayList = [(420, 559), (420, 961), (1244, 559), (1244, 961)]
fontFace = cv2.FONT_HERSHEY_SIMPLEX

fp1 = np.float32(np.array([[[36.,135.]],[[294.,80.]],[[122.,185.]],[[224.,70.]],[[172.,206.]],[[134.,158.]],[[60.,142.]],[[80.,170.]],[[95.,218.]],[[27.,218.]],[[45.,186.]],
	[[28.,264.]],[[354., 69.]],[[230.,43.]]]))
fpc1 = []
fp2 = np.float32(np.array([[[536,125]],[[566,136]],[[574,83]],[[579,132]],[[539,95]],[[513,173]],[[467,177]],[[584,77]],[[80,106]],[[108,118]],[[161,87]]]))
fpc2 = [[119,0,[[163,118],[142,163],[133,172]],[370,-2,[205,196],[206,210],[523,245]]],[500,3,[[33,134],[20,167],[397,268],[454,98],[625,102],[97,110]]]]
fp3 = np.float32(np.array([[[56,98]],[[127,139]],[[250, 178]],[[208, 175]],[[248,98]],[[343,160]],[[141,98]],[[145,194]],[[283,96]],[[222,273]]]))
fpc3 = [[350,0,[[355,275],[521,275],[607,196]]]]
fp4 = np.float32(np.array([[[618,227]],[[550,286]],[[535,103]],[[534,197]],[[396,117]],[[396,278]],[[378,107]],[[342,80]],[[242,270]]]))
fpc4 = [[72,-4,[[308,166],[234,104],[196,152],[284,169],[165,279],[366,281],[314,182],[191,203],[260,103],[254,272],[383,103],[492,277],[590,120]]],
		[137,3,[[529,274],[508,103],[431,106],[245,249],[242,275]]],[370,100,[[318,283],[403,287],[216,249],[195,277],[289,139],[80,143],[98,281],[89,102],[219,100],[278,100]]]]
fp5 = np.float32(np.array([[[189,158]],[[456,164]],[[414,194]],[[227,180]],[[198,182]],[[447,178]],[[215,155]],[[495,164]]]))
fpc5 = []
fp6 = np.float32(np.array([[[457,191]],[[440,314]],[[579,171]],[[380,185]],[[587,292]],[[378,295]],[[370,308]],[[560,174]]]))
fpc6 = []
fp7 = np.float32(np.array([[[188,314]],[[18,322]],[[214,239]],[[132,336]],[[44,176]],[[116,177]],[[83,235]],[[41,347]]]))
fpc7 = [[500,4,[[208,140],[392,140],[491,308],[377,340]]]]

cp1 = np.float32(np.array([[[196, 62]],[[440, 137]],[[204, 290]],[[48, 88]]]))
cp2 = np.float32(np.array([[[573.,92.]],[[597.,266.]],[[0.,194.]],[[354.,82.]]]))
cp3 = np.float32(np.array([[[165,194]],[[441,187]],[[494,282]],[[166,288]]]))
cp4 = np.float32(np.array([[[307,153]],[[670,166]],[[660,294]],[[243,270]]]))
cp5 = np.float32(np.array([[[426,166]],[[568,252]],[[103,257]],[[209,164]]]))
cp6 = np.float32(np.array([[[205,223]],[[545,232]],[[541,348]],[[144,335]]]))
cp7 = np.float32(np.array([[[91,235]],[[403,226]],[[458,329]],[[90,343]]]))
#0-full, #1-right(near) #2-left(far)
cplist = [[0,cp1],[0,cp2],[1,cp3],[2,cp4],[1,cp5],[2,cp6],[1,cp7]]
fplist = [fp1,fp2,fp3,fp4,fp5,fp6,fp7]
fpclist = [fpc1,fpc2,fpc3,fpc4,fpc5,fpc6,fpc7]
explist = [747,0,0,876,1170,0,1010]

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

def smoothList(lst,degree=21):  #degree must be odd
	width = degree / 2
	smoothed=[0]*len(lst) 
	weights = np.transpose(cv2.getGaussianKernel(degree,0))[0].tolist()
	padded = [lst[0]]*width + lst + [lst[len(lst)-1]]*width
	for i in range(len(smoothed)):  
		smoothed[i]=np.dot(padded[i:i+degree], weights)
	return smoothed

def smoothPlayerPosData(data):
	player1X = smoothList([ele[0] for ele in data[0]])
	player1Y = smoothList([ele[1] for ele in data[0]])
	player2X = smoothList([ele[0] for ele in data[1]])
	player2Y = smoothList([ele[1] for ele in data[1]])
	player3X = smoothList([ele[0] for ele in data[2]])
	player3Y = smoothList([ele[1] for ele in data[2]])
	player4X = smoothList([ele[0] for ele in data[3]])
	player4Y = smoothList([ele[1] for ele in data[3]])

	for index in range(0, len(data[0])):
		data[0][index][0] = player1X[index]
		data[0][index][1] = player1Y[index]
		data[1][index][0] = player2X[index]
		data[1][index][1] = player2Y[index]
		data[2][index][0] = player3X[index]
		data[2][index][1] = player3Y[index]
		data[3][index][0] = player4X[index]
		data[3][index][1] = player4Y[index]

	return data
def findHomoMatrixTopDown(cornerPts, fieldtype):
	# if having corners points, then do:
	# src = cornerPts
	src = cornerPts

	# original corner pts
	# original corner pts
	if fieldtype == 0:
		dst = np.array([
				[70, 60],
				[470, 60],
				[470, 260],
				[70, 260]], dtype = "float32")
	elif fieldtype == 1:
		dst = np.array([
				[270, 60],
				[470, 60],
				[470, 260],
				[270, 260]], dtype = "float32")
	else: #2
		dst = np.array([
				[70, 60],
				[270, 60],
				[270, 260],
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

		startOutOfView = False
		firstpos = []
		print count
		print len(playerPosList[playerIndex])
		for frameIndex in range(count):
			homoMatrix = homoMatrixList[frameIndex]
			temp = playerPosList[playerIndex][frameIndex]
			if temp[0] == -1: #out of view
				if frameIndex == 0:
					startOutOfView = True
					mappedPlayerPosList[playerIndex][frameIndex] = [-1,-1]
				else:
					if startOutOfView == False:
						mappedPlayerPosList[playerIndex][frameIndex] = list(mappedPlayerPosList[playerIndex][frameIndex-1])
			else:
				if startOutOfView:
					firstpos = limitRegion(cv2.perspectiveTransform(np.float32(np.asarray([[temp]])), homoMatrix).tolist()[0][0], playerIndex)
					startOutOfView = False
				mappedPlayerPosList[playerIndex][frameIndex] = limitRegion(cv2.perspectiveTransform(np.float32(np.asarray([[temp]])), homoMatrix).tolist()[0][0], playerIndex)
		for frameIndex in range(count):
			if mappedPlayerPosList[playerIndex][frameIndex] != [-1,-1]:
				break
			else:
				mappedPlayerPosList[playerIndex][frameIndex] = firstpos

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
		if lastplay:
			print 'captured landing point!!!!!!!!!'
			lastBallPts.append([index,x,y])
		else:
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
		if not ready:
			ready = True
			initpos = [-1,-1]
			print 'starting out of view!'
			print 'halted: ' + str(halted)
			halted = True
		else:
			halted = not halted
			print 'halted: ' + str(halted)
	#if event == cv2.EVENT_MOUSEMOVE:
	if ready and not saved:
		saved = True
		if not halted:
			mouselist.append([x,y])	
			print [x,y]	

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

		cap = cv2.VideoCapture('beachVolleyball' + str(video+1) + '.mov')
		_, frame = cap.read()
		#mask = np.zeros_like(frame)

		frameCount = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		listActualSize = frameCount

		grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)				
		cornerPts = cplist[video][1]
		featurePts = fplist[video]

		curfpclist = list(fpclist[video])
		
		homoMatrix = findHomoMatrixTopDown(cornerPts, cplist[video][0])

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
		lastplay = False
		lastsave = [-1,-1]
		initpos = []
		lastBallPts = []
		if iterateIndex == 4:
			lastplay = True
		cv2.namedWindow('frame')
		cv2.setMouseCallback('frame', on_mouse)

		while(_):
			_, frame = cap.read()
			if explist[video] == index or not _:
				print index
				for i in range(index-len(mouselist)):
					mouselist = [initpos] + mouselist
				print len(mouselist)
				break

			#cv2.namedWindow('frame')
			saved = False
			#cv2.setMouseCallback('frame', on_mouse)
			if halted:
				mouselist.append([-1,-1])
				print mouselist[-1]
			else:
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
			homoMatrix = findHomoMatrixTopDown(curr, cplist[video][0])

			homoMatrixList.append(homoMatrix.copy())
			frameList.append(grayOld)
			grayOld = grayNew.copy()
			#complementary points
			if len(curfpclist) > 0 and index - 1 == curfpclist[0][0]:
				if curfpclist[0][1] == 100: #replace all pts
					temp = []
				else:
					if curfpclist[0][1] == 0: #append only
						temp = good_new.tolist()
					elif curfpclist[0][1] > 0: #remove front items
						temp = good_new.tolist()
						temp2 = []
						for i in range(curfpclist[0][1]):
							temp2.append(temp.pop())
						temp = []
						for i in range(curfpclist[0][1]):
							temp.append(temp2[i])
					else: #remove back items
						temp = good_new.tolist()
						for i in range(-curfpclist[0][1]):
							p = temp.pop()
				for i in range(len(curfpclist[0][2])):
					temp.append(curfpclist[0][2][i])
				good_new = np.float32(np.asarray(temp))
				curfpclist.pop(0)
			else:
				pass
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

	frameCount = frameCountCopy

	testTopViewList = []
	mappedPlayerPosList, mappedTouchList, mappedJumpList, mappedLastBallPtsList = mapRawPtsToAnime(playerPosList, 
		touchlistTotal, jumplistTotal, lastBallPtsList, homoMatrixList)
	print mappedPlayerPosList
	mappedPlayerPosList = smoothPlayerPosData(mappedPlayerPosList)
	print mappedPlayerPosList
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
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
	tdvideo = cv2.VideoWriter('topdown' + str(video+1) + '.mov',fourcc,fps=59,frameSize=(topViewArtNew.shape[1],topViewArtNew.shape[0]),isColor=1)
	stvideo = cv2.VideoWriter('stats' + str(video+1) + '.mov',fourcc,fps=59,frameSize=(boardViewNew.shape[1],boardViewNew.shape[0]),isColor=1)

	for frame in testTopViewList:
		tdvideo.write(frame)

	for frame in boardViewList:
		stvideo.write(frame)

	cv2.destroyAllWindows()
	cap.release()
	pass
