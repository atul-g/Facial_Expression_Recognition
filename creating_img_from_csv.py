import pandas as pd
import numpy as np
import cv2
import os
l=[1,1,1,1,1,1,1]
data=pd.read_csv('fer2013.csv')
pathname=os.path.abspath('.')+'/exp_images/'
for row in data.itertuples():
    img=np.asarray(row[2].split(' '), dtype=np.uint8)
    img=np.resize(img, (48, 48))
    
    if row[1]==0:
        cv2.imwrite(pathname+'Angry/'+str(l[0])+'.jpg', img[:, :, 1])

        l[0]=l[0]+1
    elif row[1]==1:
        cv2.imwrite(pathname+'Disgust/'+str(l[1])+'.jpg', img[:, :, 1])
        l[1]=l[1]+1
    elif row[1]==2:
        cv2.imwrite(pathname+'Fear/'+str(l[2])+'.jpg', img[:, :, 1)]
        l[2]=l[2]+1
    elif row[1]==3:
        cv2.imwrite(pathname+'Happy/'+str(l[3])+'.jpg', img[:, :, 1])
        l[3]=l[3]+1
    elif row[1]==4:
        cv2.imwrite(pathname+'Sad/'+str(l[4])+'.jpg', img[:, :, 1])
        l[4]=l[4]+1
    elif row[1]==5:
        cv2.imwrite(pathname+'Surprise/'+str(l[5])+'.jpg', img[:, :, 1])
        l[5]=l[5]+1
    elif row[1]==6:
        cv2.imwrite(pathname+'Neutral/'+str(l[6])+'.jpg', img[:, :, 1])
        l[6]=l[6]+1
    else:
        pass
        
print('images have been created and saved')

