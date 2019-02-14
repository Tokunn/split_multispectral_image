#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns

def split_channel(i, name_list, image):
    picked_channel = image[:,:,i]
    picked_channel3ch = np.dstack((picked_channel, picked_channel, picked_channel))
    plt.imshow(picked_channel3ch)
    #plt.show()
    pil_picked_channel3ch = Image.fromarray(picked_channel3ch)
    pil_picked_channel3ch.save('split/'+name_list[i]+'.jpg')



nir_image = np.array(Image.open('./sample/NIR.jpg'))
name_list = ['00rededge', '01none', '02nir']
for i in range(3):
    #split_channel(i, name_list, nir_image)
    pass

rgb_image = np.array(Image.open('./sample/RGB.jpg'))
name_list = ['10r', '11g', '12b']
for i in range(3):
    #split_channel(i, name_list, rgb_image)
    pass





picked_nir = nir_image[:,:,2]
picked_nir3ch = np.dstack((picked_nir,picked_nir,picked_nir))
picked_r   = rgb_image[:,:,0]
picked_r3ch = np.dstack((picked_r,picked_r,picked_r))

ndvi3ch = (picked_nir-picked_r) / (picked_nir+picked_r)
print(ndvi3ch.shape)
ndvi3ch = ndvi3ch * 255.0
ndvi3ch.dtype = 'int32'
ndvi3ch = np.nan_to_num(ndvi3ch)
print(ndvi3ch.shape)
pil_ndvi = Image.fromarray(ndvi3ch)
pil_ndvi.save('split/ndvi.png')

#plt.figure()
#sns.heatmap(ndvi)
#plt.savefig('split/ndvi.jpg')

