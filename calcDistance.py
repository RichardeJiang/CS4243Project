import cv2
import cv2.cv as cv
import numpy as np
import numpy.linalg as la

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
	playerPos = np.float32(np.array([[[133, 130]],[[111, 210]], [[297, 162]],[[536, 139]]]))

	cap = cv2.VideoCapture('panorama.mov')
	_, frame = cap.read()

	grayOld = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cumulatedDistances = [0.0, 0.0, 0.0, 0.0]

	dashBoard = cv2.imread('dashBoard.jpg')
	dashBoardOriginal = dashBoard.copy()
	height, width = dashBoard.shape[:2]
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('stats1.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
	orgList = [(480, 428), (480, 816), (1304, 428), (1304, 816)]
	fontFace = cv2.FONT_HERSHEY_SIMPLEX
	testCalcDistanceList = []

	while(_):
		_, frame = cap.read()

		if not _:
			break

		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# calculate new player position coordinates
		playerNewPos, st, err = cv2.calcOpticalFlowPyrLK(grayOld, grayNew, playerPos, None, **lk_params)

		goodOld = playerPos[st==1]
		goodNew = playerNewPos[st==1]

		distances = calculatePlayerDistancePerFrame(goodOld, goodNew)

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
		#video.write(dashBoard)
		dashBoard = dashBoardOriginal.copy()

		grayOld = grayNew.copy()
		playerPos = goodNew.reshape(-1, 1, 2)

	for board in testCalcDistanceList:
		video.write(board)

	cv2.destroyAllWindows()
	cap.release()
	pass