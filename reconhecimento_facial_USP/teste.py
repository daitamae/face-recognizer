import numpy as np

filetxt=open(r'C:\Users\User\Desktop\reconhecimento_facial_USP\pessoas.txt','r')
p=eval(filetxt.read())
if 6 in p:
    p2=p
else:
    p2=np.append(p,int(6))
print(p2)