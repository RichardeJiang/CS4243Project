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

if (__name__ == "__main__"):
	cap = cv2.VideoCapture('vid.mov')
	_, currFrame = cap.read()
	count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv.CV_CAP_PROP_FPS))
	print count, fps
	width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)) * 2
	height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	video = cv2.VideoWriter('panorama.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
	for fr in range(1, count - 1):
		_, nextFrame = cap.read()
		# grayCurrFrame = cv2.cvtColor(currFrame, cv.CV_RGB2GRAY)
		# grayNextFrame = cv2.cvtColor(nextFrame, cv.CV_RGB2GRAY)
		grayCurrFrame = currFrame
		grayNextFrame = nextFrame

		### find homography matrix

		sift = cv2.SIFT()
		currKeyPoints, currDescriptor = sift.detectAndCompute(grayCurrFrame, None)
		nextKeyPoints, nextDescriptor = sift.detectAndCompute(grayNextFrame, None)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)

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

			warped = cv2.warpPerspective(grayCurrFrame, homographyMatrix, (width, height))
			video.write(warped)

		currFrame = warped

	pass