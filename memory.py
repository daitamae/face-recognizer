import cv2
import numpy as np

camera = cv2.VideoCapture(0)
count=0
name=input('Subject name')
filetxt=open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\pessoas.txt','r+')
filecore=eval(filetxt.read())
nuspper=name
filetxt.truncate()
filetxt.close()
filetxt=open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\pessoas.txt','r+')
if nuspper in filecore:
    filecore.pop(filecore.index(nuspper))
    filecore.insert(0,nuspper)
    filetxt.write(str(filecore))
    print(str(filecore))
else:
    filecore.append(nuspper)
    filetxt.write(str(filecore))
    print(str(filecore))
nusp=filecore.index(name)
filetxt.close()
print(nusp)
face_id=nusp
while True:
    return_value,img = camera.read()
    face_cascade = cv2.CascadeClassifier(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1,5)
    faceSamples=[]
    Ids=[]
    
    for (x,y,w,h) in faces:
        count+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        Id=int(nusp)
        imgc = gray[y:y+h, x:x+w]
        faceSamples.append(imgc)
        Ids.append(Id)
        
    print(faces)
    img2=cv2.resize(img,None,fy=1.2,fx=1.2,interpolation=cv2.INTER_AREA)
    cv2.imshow('img',img2)    
    if cv2.waitKey(1)& 0xFF == 27:
        break
    elif count>100:
        break
camera.release()
cv2.destroyAllWindows()

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Train the model using the faces and IDs
recognizer.train(faceSamples, np.array(Ids))

# Save the model into trainer.yml
recognizer.save(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\memory\trainer_'+name+'_.yml')

print(name+"'s facial characteristics saved")

