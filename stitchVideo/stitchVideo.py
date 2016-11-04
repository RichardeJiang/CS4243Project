from pyimagesearch.panorama import Stitcher
import imutils
import cv2
import numpy as np

filenames = ['beachVolleyball1.mov', 'beachVolleyball2.mov', 'beachVolleyball3.mov', 'beachVolleyball4.mov', 'beachVolleyball5.mov', 'beachVolleyball6.mov', 'beachVolleyball7.mov'] 
ofile = ['bvout1.avi', 'bvout2.avi', 'bvout3.avi', 'bvout4.avi', 'bvout5.avi', 'bvout6.avi', 'bvout7.avi', ]

for i in range(0, len(filenames)):
	print "process", filenames[i]
	cap = cv2.VideoCapture(filenames[i])
	print "Frame Count:", cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	print "Frame per second:", cap.get(cv2.cv.CV_CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))

	w=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	h=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(ofile[i], fourcc, fps, (w*3, 2*h))
	ret, old_frame = cap.read()

	x,y,z = old_frame.shape
	bg = np.zeros(shape=(2*x, 3*y, z), dtype=np.uint8)
	bg[x/2:x+x/2, y:2*y] = old_frame

	frame_aug = np.zeros(shape=(2*x, 3*y, z), dtype=np.uint8)
	stitcher = Stitcher()
	for fr in range(1, frame_count):
		ret, frame = cap.read()
		frame_aug.fill(0)
		frame_aug[x/2:x+x/2, y:2*y] = frame

		print fr, bg.shape
		result = stitcher.stitch([frame_aug, bg])
		out.write(result)
		
		cv2.imshow("out", result)
		bg = result.copy()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	out.release()
	cv2.destroyAllWindows()
	cap.release()

