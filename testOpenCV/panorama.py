import cv2
import cv2.cv as cv
import numpy as np

def detectFeatures(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	detector = cv2.FeatureDetector_create("SIFT")
	kps = detector.detect(gray)

	# extract features from the image
	extractor = cv2.DescriptorExtractor_create("SIFT")
	(kps, features) = extractor.compute(gray, kps)
	kps = np.float32([kp.pt for kp in kps])
	return kps, features

def matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio = 0.75, threshold = 0.4):
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	matches = []
	
	return

def findHomographyMatrix(ptFrame1, ptFrame2):
	# ptFrame1 is 3D numpy array, each element is 2D array of the coordinate
	if (len(ptFrame1) < 4 or len(ptFrame2) < 4):
		return
	else:
		homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		return homographyMatrix

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph matrix H'''
    height1,width1 = img1.shape[:2]
    height2,width2 = img2.shape[:2]
    p1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
    p2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
    p3 = cv2.perspectiveTransform(p2, H)

    #combint two images and reshape the image size
    points = np.concatenate((p1, p3), axis=0)
    [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
    M = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

    outputImg = cv2.warpPerspective(img2, M.dot(H), (xmax-xmin, ymax-ymin))
    outputImg[-ymin:height1-ymin,-xmin:width1-xmin] = img1
    return outputImg

if (__name__ == "__main__"):
	cap = cv2.VideoCapture('vid.mov')
	_, currFrame = cap.read()
	count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv.CV_CAP_PROP_FPS))
	print count, fps
	# width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)) * 2
	# height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	
	index = 0
	results = []

	sift = cv2.SURF(400)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	for fr in range(1, count - 1):
		_, nextFrame = cap.read()
		# grayCurrFrame = cv2.cvtColor(currFrame, cv.CV_RGB2GRAY)
		# grayNextFrame = cv2.cvtColor(nextFrame, cv.CV_RGB2GRAY)
		grayCurrFrame = currFrame
		grayNextFrame = nextFrame

		### find homography matrix

		currKeyPoints, currDescriptor = sift.detectAndCompute(grayCurrFrame, None)
		nextKeyPoints, nextDescriptor = sift.detectAndCompute(grayNextFrame, None)


		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(currDescriptor, nextDescriptor, k=2)

		goodMatches = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				goodMatches.append(m)

		if len(goodMatches) >= 4:
			src_pts = np.float32([ currKeyPoints[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
			dst_pts = np.float32([ nextKeyPoints[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

			homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

			#warped = cv2.warpPerspective(grayCurrFrame, homographyMatrix, (width, height))
			warped = warpTwoImages(grayNextFrame, grayCurrFrame, homographyMatrix)
			#cv2.imwrite('a/' + str(index) + '.jpg', warped)
			results.append(warped)
			#video.write(warped)

		currFrame = warped
		index += 1
	width = len(results[0][0])
	height = len(results[0])
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('panorama.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
	for warp in results:

		video.write(warp)
	pass