filetxt=open(r'C:\Users\User\Desktop\reconhecimento_facial_USP\pessoas.txt','r+')
filecore=filetxt.read()
file=eval(filecore)
camera = cv2.VideoCapture(0)
count=0
name=input('Subject name')
nuspper=input('NUSP')
if str(nuspper) in file:
    nusp=file
    print(nusp)
else:
    nusp=np.append(file,int(nuspper))
    print(nusp)
    
filetxt.truncate()
print(filetxt.read()+'AAA')
filetxt.close()
filetxt=open(r'C:\Users\User\Desktop\reconhecimento_facial_USP\pessoas.txt','r+')
filetxt.write(str(nusp))
print(filetxt.read())
filetxt.close()
