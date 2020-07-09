
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
from PIL import Image
from scipy.misc import imresize
from random import randrange
import numpy as np
address='./boatman_image/'
start_num=345
end_num=856
pic_video=8
ver_num=256
hor_num=256
picture_num=end_num-start_num+1
video_num=picture_num-7
orig1=np.zeros((ver_num,hor_num,picture_num), dtype = None, order = 'C')
orig=np.zeros((ver_num,hor_num,video_num*pic_video), dtype = None, order = 'C')
#open(newaddress,"w+") 
def Normalize(data):
    range1 = np.max(data) - np.min(data)
    return (data - np.min(data)) / range1    
num=0
C = [[randrange(0,2,1) for e in range(256)] for e in range(256)]
for i in range(start_num,end_num+1):
    address1=address+str(i)
    address2=address1+'.bmp'    
    orig1[:,:,num]=Normalize(mpimg.imread(address2))
    num +=1
code=np.zeros((ver_num,hor_num,pic_video),dtype = None, order = 'C')
for i in range(pic_video):
    code[:,:,i]=np.roll(C,i,axis=0)
E=np.zeros((ver_num,hor_num,video_num), dtype = None, order = 'C')
for i in range(video_num):
    for j in range(pic_video):
        orig[:,:,i*pic_video+j]=np.roll(orig1[:,:,i+j],j,axis=0)
    B=np.multiply(orig[:,:,i*pic_video:(i+1)*pic_video],code)
    E[:,:,i]=np.sum(B,axis=2)
stepsize=1
truth=orig
sio.savemat('./train/boatman/Data.mat',{'E':E,'stepsize':stepsize,'truth':orig})
sio.savemat('./train/boatman/Code.mat',{'code':code})
video_num=picture_num-7  
    

