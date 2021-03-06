import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from glob import glob
from tqdm import tqdm
import os

nir_dir = './sample/NIR'
rgb_dir = './sample/RGB'
output_dir = './split'
os.makedirs(output_dir, exist_ok=True)

def split_channel(i, name_list, image, nir_name):
    os.makedirs(os.path.join('split', os.path.basename(nir_name)), exist_ok=True)
    picked_channel = image[:,:,i]
    picked_channel3ch = np.dstack((picked_channel, picked_channel, picked_channel))
    #plt.imshow(picked_channel3ch)
    #plt.show()
    pil_picked_channel3ch = Image.fromarray(picked_channel3ch)
    pil_picked_channel3ch.save(os.path.join('split', os.path.basename(nir_name), name_list[i]+'.jpg'))
    return pil_picked_channel3ch

def split(nir_name, rgb_name, output_name):
    img_list = {}
    nir_image = np.array(Image.open(nir_name))
    name_list = ['00rededge', '01none', '02nir']
    #plt.figure()
    for i in range(3):
        img = split_channel(i, name_list, nir_image, nir_name)
        img_list[name_list[i]] = img
        #plt.subplot(2,3,i+1)
        #plt.imshow(img)

    rgb_image = np.array(Image.open(rgb_name))
    name_list = ['10r', '11g', '12b']
    for i in range(3):
        img = split_channel(i, name_list, rgb_image, rgb_name)
        img_list[name_list[i]] = img
        #plt.subplot(2,3,i+4)
        #plt.imshow(img)
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

    plt.figure(figsize=(40,30))
    sns.heatmap(ndvi, cbar=False, square=True)
    #plt.show()
    # plt.box(on=None)
    plt.axis('off')
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    plt.savefig(output_name)


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

def main():
    for nir_name, rgb_name in tqdm(zip(sorted(glob(os.path.join(nir_dir, '*.jpg'))), sorted(glob(os.path.join(rgb_dir, '*.jpg'))))):
        output_name = os.path.join(output_dir, os.path.basename(nir_name), 'NDVI.png')
        split(nir_name, rgb_name, output_name)

if __name__ == '__main__':
    main()
