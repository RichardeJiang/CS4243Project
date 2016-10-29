import sys
import cv2
import cv2.cv as cv
import numpy as np

cap = cv2.VideoCapture('beachVolleyball1.mov')
width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

print "frame width = ",width
print "frame height = ",height
print "frame count = ",frameCount

for fr in range(1, frameCount):
	_,img = cap.read()
	if not _:
		sys.exit(0)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,5,3,0.04)
	dst = cv2.dilate(dst,None)
	img[(dst>0.01*dst.max()) & (dst < 0.5*dst.max())]=[0,0,255]
	cv2.imshow('dst',img)	
	print "fr= ",fr
	cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

cv2.imwrite('background.png',normImg)


