import cv2
import numpy as np

def topDownView(cornerPts, playerPts):
	image = cv2.imread('test.jpg')

	maxWidth = 600
	maxHeight = 300
	# if having corners points, then do:
	# src = cornerPts
	src = np.array([
			[130, 550],
			[1630, 575],
			[1760, 860],
			[0, 800]], dtype = "float32")

	dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	mappedPlayerPos = []
	for pts in playerPts:
		pts = np.asarray(pts.tolist().append(1))
		newPos = M.dot(pts).tolist()
		newPos = [ele/newPos[2] for ele in newPos]
		mappedPlayerPos.append(newPos[:2])

	mappedPlayerPos = np.asarray(mappedPlayerPos)
	return

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
