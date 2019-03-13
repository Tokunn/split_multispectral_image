#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns

def split_channel(i, name_list, image):
    picked_channel = image[:,:,i]
    picked_channel3ch = np.dstack((picked_channel, picked_channel, picked_channel))
    #plt.imshow(picked_channel3ch)
    #plt.show()
    pil_picked_channel3ch = Image.fromarray(picked_channel3ch)
    pil_picked_channel3ch.save('split/'+name_list[i]+'.jpg')
    return pil_picked_channel3ch


img_list = {}
nir_image = np.array(Image.open('./sample/g_NIR.jpg'))
name_list = ['00rededge', '01none', '02nir']
#plt.figure()
for i in range(3):
    img = split_channel(i, name_list, nir_image)
    img_list[name_list[i]] = img
    plt.subplot(2,3,i+1)
    plt.imshow(img)

rgb_image = np.array(Image.open('./sample/g_RGB.jpg'))
name_list = ['10r', '11g', '12b']
for i in range(3):
    img = split_channel(i, name_list, rgb_image)
    img_list[name_list[i]] = img
    plt.subplot(2,3,i+4)
    plt.imshow(img)
#plt.show()


ir = np.asarray(img_list["02nir"], dtype=np.float32)
#ir = np.asarray(img_list["00rededge"], dtype=np.float32)
ir[:,:,0][ir[:,:,0]==0] = np.nan
ir[:,:,1][ir[:,:,1]==0] = np.nan
ir[:,:,2][ir[:,:,2]==0] = np.nan
red = np.asarray(img_list["10r"], dtype=np.float32)
red[:,:,0][red[:,:,0]==0] = np.nan
red[:,:,1][red[:,:,1]==0] = np.nan
red[:,:,2][red[:,:,2]==0] = np.nan

nomi = ir[:,:,0] - red[:,:,0]
domi = ir[:,:,0] + red[:,:,0]

ndvi = domi.copy()
index = domi != 0
ndvi[:] = np.nan
ndvi[index] = nomi[index]/domi[index]
del nomi, domi

#plt.figure()
#ax = plt.gca()
#im = ax.imshow(ndvi, vmin=0, vmax=1, cmap=plt.cm.jet)
#plt.colorbar(im)
#plt.show()
#ndvi[np.isnan(ndvi)] = -9999

plt.figure()
sns.heatmap(ndvi)
plt.show()
plt.savefig('split/ndvi.jpg')


#picked_nir = nir_image[:,:,2]
#picked_nir3ch = np.dstack((picked_nir,picked_nir,picked_nir))
#picked_r   = rgb_image[:,:,0]
#picked_r3ch = np.dstack((picked_r,picked_r,picked_r))
#
#ndvi3ch = (picked_nir-picked_r) / (picked_nir+picked_r)
#print(ndvi3ch.shape)
#ndvi3ch = ndvi3ch * 255.0
#ndvi3ch.dtype = 'int32'
#ndvi3ch = np.nan_to_num(ndvi3ch)
#print(ndvi3ch.shape)
#pil_ndvi = Image.fromarray(ndvi3ch)
#pil_ndvi.save('split/ndvi.png')
#
##plt.figure()
##sns.heatmap(ndvi)
##plt.savefig('split/ndvi.jpg')
#
