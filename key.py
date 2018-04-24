import os

file=open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\name.txt','r')
name=file.read()
file.close()
print(name+'\n')
open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\dados\{}.txt'.format(name),'w+')
secretfile=open(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\dados\{}.txt'.format(name),'r+')
if (secretfile.read())=='':
    secretfile.write('bem vindo,'+name+'!')
os.startfile(r'C:\Users\Dai\Documents\Python Scripts\reconhecimento_facial_USP\dados\{}.txt'.format(name))
