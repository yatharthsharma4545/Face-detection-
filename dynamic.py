import cv2,time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0) # create VideoCapture object..it triggers the camera
a=1 #a count variable for no. of frames

while True :
    a+=1  #count to count no. of frames
    check ,frame=video.read()   # a method of VideoCapture which gives check of image:-T/F and a numpy array:-it represents the first image that video captures
    gray = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY)   #covert the rgb image to grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)   #a method of CascadeClassifier to detect co-ordinates of faces ....

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow('dynamic face detection',frame)  #show the frame on console 'dynamic face detection'
    key = cv2.waitKey(1)  #changes the frame after 1 milisec.

    if key==ord("q"):   #if q is press then break out of loop
        break

print(a)    #print frames
video.release()  #closes the window 'dynamic face detection'
cv2.destroyAllWindows() # De-allocate any associated memory usage
