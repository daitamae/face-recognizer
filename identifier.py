import cv2

count=0
count2=0
count3=0
file=open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\name.txt','r+')
file.truncate()
file.close()
file=open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\name.txt','r+')
name=input('dê o seu nome')
file.write(name)
recognizer = cv2.face.LBPHFaceRecognizer_create()

cascadePath = r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True and count3<1:
    ret, im2 =cam.read()
    im=cv2.resize(im2,None,fy=1.3,fx=1.3,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.08,5)
    recognizer.read(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\memory\trainer_'+name+'_.yml')
    if count<100:
        for(x,y,w,h) in faces:
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (255,0,0), 4)
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print(Id,confidence)
            print((100 - confidence))
            if (round(100 - confidence)>50):
                phrase = name+' {0:.2f}%'.format(round(100 - confidence, 2))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,0,0), -1)
                cv2.putText(im, str(phrase), (x,y-40), font, 1, (255,255,255), 3)
                key='processing image...'
                cv2.rectangle(im, (x-22,y-130), (x+w+22, y-90), (255,0,0), -1)
                cv2.putText(im, str(key), (x,y-90), font, 1, (255,255,255), 3)
                cv2.putText(im, str(count)+'%',(x+300,y-40), font, 1, (255,255,255), 3)
                count+=2
            else:
                cv2.rectangle(im, (x-40,y-90), (x+w+40, y-22), (0,0,255), -1)
                cv2.putText(im,'not recognized', (x-30,y-40), font, 1, (255,255,255), 3)
                continue
    else:
        for(x,y,w,h) in faces:
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print(Id,confidence)
            print((100 - confidence))
            if (round(100 - confidence)>50):
                phrase = name+' {0:.2f}%'.format(round(100 - confidence, 2))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                cv2.putText(im, str(phrase), (x,y-40), font, 1, (255,255,255), 3)
                key='permission given'
                cv2.rectangle(im, (x-22,y-130), (x+w+22, y-90), (0,255,0), -1)
                cv2.putText(im, str(key), (x,y-90), font, 1, (255,0,0), 3)
                if count2<20:
                    count2+=1
                else:
                    exec(open('key.py').read())
                    cam.release()
                    cv2.destroyAllWindows()
                    count3+=1
            else:
                cv2.rectangle(im, (x-40,y-90), (x+w+40, y-22), (0,0,255), -1)
                cv2.putText(im,'not recognized', (x-30,y-40), font, 1, (255,255,255), 3)
                continue
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()
