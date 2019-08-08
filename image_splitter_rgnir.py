import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from glob import glob
from tqdm import tqdm
import os

input_dir = './sample/RGNIR'
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

def split(name, output_name):
    img_list = {}
    image = np.array(Image.open(name))
    name_list = ['00r', '01g', '02nir']
    #plt.figure()
    for i in range(3):
        img = split_channel(i, name_list, image, name)
        img_list[name_list[i]] = img
        #plt.subplot(2,3,i+1)
        #plt.imshow(img)


    ir = np.asarray(img_list["02nir"], dtype=np.float32)
    #ir = np.asarray(img_list["00rededge"], dtype=np.float32)
    ir[:,:,0][ir[:,:,0]==0] = np.nan
    ir[:,:,1][ir[:,:,1]==0] = np.nan
    ir[:,:,2][ir[:,:,2]==0] = np.nan
    red = np.asarray(img_list["00r"], dtype=np.float32)
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

def main():
    for name in tqdm(sorted(glob(os.path.join(input_dir, '*.JPG')))):
        output_name = os.path.join(output_dir, os.path.basename(name), 'NDVI.png')
        split(name, output_name)

if __name__ == '__main__':
    main()
