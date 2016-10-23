import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

w=int(cap.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH ))
h=int(cap.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT ))

#fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc = cv.cv.CV_FOURCC(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (w, h))

while(cap.isOpened()):
	ret, frame = cap.read()

	if (ret == True):
		frame = cv.flip(frame, 0)

		out.write(frame)
		cv.imshow('frame', frame)

		if (cv.waitKey(1) & 0xFF == ord('q')):
			break

	else:
		break

cap.release()
out.release()
cv.destroyAllWindows()