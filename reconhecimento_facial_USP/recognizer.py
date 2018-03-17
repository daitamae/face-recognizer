import cv2
import numpy as np
import os
import glob

filetxt=open(r'C:\Users\User\Desktop\reconhecimento_facial_USP\pessoas.txt','r')
nusp=input('NUSP')
recognizer = cv2.face.LBPHFaceRecognizer_create()

for file in glob.glob(r'C:\Users\User\Desktop\reconhecimento_facial_USP\memory\*.yml'):
    recognizer.read(file)

cascadePath = r'C:\Users\User\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(Id,confidence)
        print((100 - confidence))
        if(Id == 10705620):
            if (round(100 - confidence)>50):
                Id = 'Daisuke {0:.2f}%'.format(round(100 - confidence, 2))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)
                key='permission given'
                cv2.rectangle(im, (x-22,y-130), (x+w+22, y-90), (0,255,0), -1)
                cv2.putText(im, str(key), (x,y-90), font, 1, (255,0,0), 3)
            elif(round(100 - confidence)<=50):
                key='permission denied'
                cv2.rectangle(im, (x-22,y-130), (x+w+22, y-90), (0,255,0), -1)
                cv2.putText(im, str(key), (x,y-90), font, 1, (0,0,255), 3)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()