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

    hog = cv.HOGDescriptor()
    hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )
    # hog.setSVMDetector( cv.HOGDescriptor_getPeopleDetector48x96() )
    cap=cv.VideoCapture('vid.mov')
    while True:
        _,frame=cap.read()
        print np.shape(frame)
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame,found)
        cv.imshow('human',frame)
        ch = 0xFF & cv.waitKey(1)
        if ch == 27:
            break
    cv.destroyAllWindows()