import cv2
from matplotlib.pyplot import contour

# refrence image
ref_img=cv2.imread('reference.png')
ref_img=cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)
ref_img=cv2.GaussianBlur(ref_img,(21,21),0)
# reference video that is webcam
video=cv2.VideoCapture("test.mp4")
# video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# video.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
    status, frame=video.read()
    # converting color to gray and decreasing noise
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    # difference between image and video
    diff=cv2.absdiff(ref_img,gray)
    # remove noise
    thresh=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    thresh=cv2.dilate(thresh,None,iterations=2)

    # contours
    cnts,res=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Threshold",frame)
    # cv2.imshow("Threshold",thresh)
    # cv2.imshow("Difference",diff)
    # cv2.imshow("Gray Video",gray)



    key=cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyWindow()