import numpy as np
import cv2


def findHomographyMatrix(src_pts, dst_pts):
	# ptFrame1 is 3D numpy array, each element is 2D array of the coordinate
	if (len(src_pts) < 4 or len(dst_pts) < 4):
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


cap = cv2.VideoCapture('beachVolleyball1.mov')

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
p0 = np.float32(np.array([[[36.,135.]],[[294.,80.]],[[122.,185.]],[[224.,70.]],[[172.,206.]],[[134.,158.]],[[60.,142.]],[[80.,170.]],[[206.,112.]]]))

#[[440.,138.]],[[310.,78.]],[[494.,188.]],[[202.,290.]]
print p0
#p0 = [[[240,291]],[[440,138]]]
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

pano = np.float32(np.array([[[518.,267.]],[[458.,660.]],[[141.,277.]],[[142.,168.]]]))
orin = np.float32(np.array([[[202.,288.]],[[440.,138.]],[[80.,134.]],[[282.,86.]]]))
pimg = cv2.imread('pano.png')
curr = orin
count = 0
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    hi = findHomographyMatrix(good_old, good_new)
    curr = cv2.perspectiveTransform(curr, hi)
    hg = findHomographyMatrix(curr, pano)
    fimg = warpTwoImages(pimg, frame, hg)
    cv2.imwrite('fimg'+ str(count) + '.png', fimg)
    #cv2.imshow('frame',fimg)
    #k = cv2.waitKey(1000) & 0xff
    #if k == 27:
    #    break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    count += 1

cv2.destroyAllWindows()
cap.release()
