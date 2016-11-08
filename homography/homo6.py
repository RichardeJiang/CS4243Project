import numpy as np
import cv2

loy = 335.
uoy = 20.
lox = 144.
uox = 1100.


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
    points = np.concatenate((p1, p3), axis=0)
    [xmin, ymin] = [xcut-lox, ycut-loy]
    [xmax, ymax] = [xcut+uox, ycut+uoy]
    M = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
    outputImg = cv2.warpPerspective(img2, M.dot(H), (int(xmax-xmin), int(ymax-ymin)))
    return outputImg


cap = cv2.VideoCapture('beachVolleyball6.mov')

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
p0 = np.float32(np.array([[[457,191]],[[440,314]],[[579,171]],[[380,185]],[[587,292]],[[378,295]],[[370,308]],[[560,174]]]))
print p0
mask = np.zeros_like(old_frame)

cutp = np.float32(np.array([[[144,335]]]))
pano = np.float32(np.array([[[541,348]],[[545,232]],[[205,223]],[[144,335]]]))
orin = np.float32(np.array([[[541,348]],[[545,232]],[[205,223]],[[144,335]]]))
pimg = cv2.imread('pano.png')
curr = orin
count = 0
results = []

while(ret):
    ret,frame = cap.read()
    print count
    if (not ret):
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    hi = findHomographyMatrix(good_old, good_new)
    curr = cv2.perspectiveTransform(curr, hi)
    cutp = cv2.perspectiveTransform(cutp, hi)
    hg = findHomographyMatrix(curr, pano)
    fimg = warpTwoImages(pimg, frame, hg, cutp)
    results.append(fimg)
    #if count % 100 == 0:
        #cv2.imwrite('fimgcut'+ str(count) + '.png', fimg)
    #cv2.imshow('frame',fimg)
    #if count == 350:
    #    cv2.imwrite('350.png',frame)
    #print count
    #k = cv2.waitKey(1) & 0xff
    #if k == 27:
    #    break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    count += 1

height,width = fimg.shape[:2]
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
video = cv2.VideoWriter('panoramadiagonal6.mov',fourcc,fps=59,frameSize=(width,height),isColor=1)
for warp in results:
    video.write(warp)

cv2.destroyAllWindows()
cap.release()
