#import libraries of python opencv
import cv2
import numpy as np

#create VideoCapture object and read from video file
cap = cv2.VideoCapture('cars.mp4')
#use trained cars XML classifiers
car_cascade = cv2.CascadeClassifier('cars.xml')
current = 0
prev = 0
count = 0
#read until video is completed
while True:
    #capture frame by frame
    ret, frame = cap.read()
    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    #to draw arectangle in each cars 
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  
        current = current + 1    
    if (current>prev):
        count = count + (current- prev)
    cv2.putText(frame,'Count: %d' %count,(10,500), FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
    prev = current
    #display the resulting frame
    cv2.imshow('video', frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
