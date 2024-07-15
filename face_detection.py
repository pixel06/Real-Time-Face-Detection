import cv2 as cv
haar=cv.CascadeClassifier('haar_face_alt_tree.xml')
capture=cv.VideoCapture(0)
while True:
    ret,frame=capture.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face=haar.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=1)
    print(f"Number of faces={len(face)}")
    for(x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)
    cv.imshow('Detected',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
            break
capture.release()
cv.destroyAllWindows()