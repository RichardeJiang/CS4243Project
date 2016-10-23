import cv2 as cv
import numpy as np

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    # hog = cv.HOGDescriptor()
    # hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )
    redUpper = (255, 0, 0)
    redLower = (224, 91, 61)
    cap=cv.VideoCapture('vid.mov')
    while True:
        _,frame=cap.read()
        print np.shape(frame)
        # found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        # draw_detections(frame,found)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, redLower, redUpper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None
     
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
     
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
     

        cv.imshow('human',frame)
        ch = 0xFF & cv.waitKey(1)
        if ch == 27:
            break
    cv.destroyAllWindows()