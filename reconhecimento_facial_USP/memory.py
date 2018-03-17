import cv2
import numpy as np
from PIL import Image
import os

camera = cv2.VideoCapture(0)
count=0
name=input('Subject name')
nusp=input('NUSP')

face_id=nusp
while True:
    return_value,img = camera.read()
    face_cascade = cv2.CascadeClassifier(r'C:\Users\User\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05,5)
    faceSamples=[]
    Ids=[]
    
    for (x,y,w,h) in faces:
        count+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite(r"C:\Users\User\Desktop\reconhecimento_facial_USP\trainer\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        Id=int(nusp)
        imgc = gray[y:y+h, x:x+w]
        faceSamples.append(imgc)
        Ids.append(Id)
        
    print(faces)
    img2=cv2.resize(img,None,fy=0.8,fx=0.8,interpolation=cv2.INTER_AREA)
    cv2.imshow('img',img2)    
    if cv2.waitKey(1)& 0xFF == 27:
        break
    elif count>30:
        break
camera.release()
cv2.destroyAllWindows()

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Train the model using the faces and IDs
recognizer.train(faceSamples, np.array(Ids))

# Save the model into trainer.yml
recognizer.save(r'C:\Users\User\Desktop\reconhecimento_facial_USP\memory\trainer_'+name+'_.yml')

print(name+"'s facial characteristics saved")

