import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.VideoCapture("elon.jpg")

check, frame = image.read() # a method of VideoCapture which gives check of image:-T/F and a numpy array:-it represents the first image that video captures
gray = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY)  #covert the rgb image to grayscale image

# Detects faces of different sizes in the input image
faces = face_cascade.detectMultiScale(gray, 1.2, 5)

for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
# resized_image=cv2.resize( frame , (int(frame.shape[1]/2),int(frame.shape[0]/2)) )
resized_image=cv2.resize( frame ,(300,300))
cv2.imshow('static face detection',resized_image) #show the frame on console 'static face detection'
k = cv2.waitKey(0) #holds the 'static face detection' window on screen untill cancelled

image.release() # Close the window
cv2.destroyAllWindows() # De-allocate any associated memory usage
