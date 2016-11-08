import numpy as np
import cv2

loy = 200.
uoy = 96.
lox = 16.
uox = 620.


def findHomographyMatrix(src_pts, dst_pts):
	# ptFrame1 is 3D numpy array, each element is 2D array of the coordinate
	if (len(src_pts) < 4 or len(dst_pts) < 4):
		return
	else:
		homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		return homographyMatrix

def warpTwoImages(img1, img2, H, cutp):
	'''warp img2 to img1 with homograph matrix H'''
	height1,width1 = img1.shape[:2]
	height2,width2 = img2.shape[:2]
	p1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
	p2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
	p3 = cv2.perspectiveTransform(p2, H)
	height3,width3 = p3.shape[:2]
	pp = cv2.perspectiveTransform(cutp, H)
	xcut = pp[0,0,0]
	ycut = pp[0,0,1] 
	#T = np.float32([[1,0,lox - xcut],[0,1,uoy - ycut]])
	#dst = cv2.warpAffine(pp, T, (height3+max(int(lox-xcut),0),width3+max(int(uoy-ycut),0)))   
	#combint two images and reshape the image size
	points = np.concatenate((p1, p3), axis=0)
	
	[xmin, ymin] = [xcut-lox, ycut-loy]
	[xmax, ymax] = [xcut+uox, ycut+uoy]
	#[xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
	#[xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
	M = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

	outputImg = cv2.warpPerspective(img2, M.dot(H), (int(xmax-xmin), int(ymax-ymin)))
	return outputImg
	#outputImg[-ymin:height1-ymin,-xmin:width1-xmin] = img1


cap = cv2.VideoCapture('beachVolleyball2.mov')

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

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print p0
#p0 = np.float32(np.array([[[536.,125.]],[[566.,136.]],[[574.,83.]],[[579.,132.]],[[539.,95.]]]))
p0 = np.float32(np.array([[[536,125]],[[566,136]],[[574,83]],[[579,132]],[[539,95]],[[513,173]],[[467,177]],[[584,77]],[[80,106]],[[108,118]],[[161,87]]]))
#[[440.,138.]],[[310.,78.]],[[494.,188.]],[[202.,290.]]
print p0
#p0 = [[[240,291]],[[440,138]]]
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

cutp = np.float32(np.array([[[16.,200.]]]))
pano = np.float32(np.array([[[16.,200.]],[[619.,279.]],[[348.,82.]],[[574.,92.]]]))
orin = np.float32(np.array([[[0.,194.]],[[597.,266.]],[[354.,82.]],[[573.,92.]]]))
pimg = cv2.imread('pano.png')
curr = orin
count = 0
results = []

while(ret):
	ret,frame = cap.read()
	if (not ret):
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	#good_new = p1[st==1]
	#good_old = p0[st==1]
	#print good_new

	good_new = []
	good_old = []
	for i in range(len(p1)):
		if st[i] == 1 and err[i] < 100:
			good_new.append(p1[i,0])
			good_old.append(p0[i,0])
	good_new = np.asarray(good_new)
	good_old = np.asarray(good_old)


	hi = findHomographyMatrix(good_old, good_new)
	curr = cv2.perspectiveTransform(curr, hi)
	cutp = cv2.perspectiveTransform(cutp, hi)
	hg = findHomographyMatrix(curr, pano)
	fimg = warpTwoImages(pimg, frame, hg, cutp)
	results.append(fimg)
	#if count % 100 == 0:
		#cv2.imwrite('fimgcut'+ str(count) + '.png', fimg)
	#cv2.imshow('frame',fimg)
	if count == 120:
		cv2.imwrite('120.png', frame)
	print count
	k = cv2.waitKey(1)
	#if k == 27:
	#	break

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)
	if count == 119:
		temp = good_new.tolist()
		temp.append([163,118])
		temp.append([142,163])
		temp.append([133,172])
		good_new = np.float32(np.asarray(temp))
	if count == 370:
		temp = good_new.tolist()
		temp.pop()
		temp.pop()
		temp.append([205,196])
		temp.append([206,210])
		temp.append([523,245])
		good_new = np.float32(np.asarray(temp))
	if count == 500:
		temp = good_new.tolist()
		p1 = temp.pop()
		p2 = temp.pop()
		p3 = temp.pop()
		temp = []
		temp.append(p1)
		temp.append(p2)
		temp.append(p3)
		temp.append([33,134])
		temp.append([20,167])
		temp.append([397,268])
		temp.append([454,98])
		temp.append([625,102])
		temp.append([97,110])
		good_new = np.float32(np.asarray(temp))
	p0 = good_new.reshape(-1,1,2)
	count += 1

height,width = fimg.shape[:2]
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
video = cv2.VideoWriter('panoramadiagonal2.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
for warp in results:
	video.write(warp)

cv2.destroyAllWindows()
cap.release()
