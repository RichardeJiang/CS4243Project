import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('beachVolleyball1.mov')

feature_params = dict( maxCorners = 20,
						qualityLevel = 0.3,
						minDistance = 7,
						blockSize = 7)

lk_params = dict( winSize = (15, 15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#p0 = np.float32(np.array([[[133, 130]],[[111, 210]], [[297, 162]],[[536, 139]]])) # for topview.mov
#p0 = np.float32(np.array([[[98, 63]],[[174.5, 60]], [[208, 99.5]],[[487.5, 207.5]],[[292.5, 44]], [[294.5, 81.5]]]))
p0 = np.float32(np.array([[[293.5, 83]],[[355, 193]],[[172, 208]],[[46.5, 137]]]))

print p0

mask = np.zeros_like(old_frame)

while(ret):
	ret, frame = cap.read()

	if not ret:
		break

	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	good_new = p1[st==1]
	good_old = p0[st==1]

	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.ravel()
		c, d = old.ravel()
		cv2.line(mask, (a, b), (c, d),color[i].tolist(), 2)
		cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
	img = cv2.add(frame, mask)

	cv2.imshow('frame', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1, 1, 2)
cv2.destroyAllWindows()
cap.release()
