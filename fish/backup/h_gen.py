import os
import glob
import matplotlib.pyplot as plt
from heatmap import to_heatmap
from heatmap import synset_to_dfs_ids
import matplotlib.image as mpimg
import numpy as np
from keras.applications.resnet50 import preprocess_input as preprocess_input_imagenet
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import scipy
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, random_rotation, flip_axis, array_to_img
from keras import backend as K
import os
import collections
from tqdm import tqdm_notebook as tq
import json
import numpy as np
import meta
from utils import *



def get_heatmaps(img, model, inception=False, no_preprocess=False):
    x = np.expand_dims(img, axis=0)
    
    
    if inception:
        x = preprocess_input_inception(x)
    elif no_preprocess:
        pass
    else:
        x = preprocess_input_imagenet(x)

    out = model.predict(x)

    s = "n02512053" # Imagenet code for "fish"
    ids = synset_to_dfs_ids(s)
    print(out.shape)
    if K.image_dim_ordering() == 'th':
        heatmap_fish = out[0,ids].sum(axis=0)
    else:
        heatmap_fish = out[0,:,:,ids]
        print(heatmap_fish.shape)
        heatmap_fish = heatmap_fish.sum(axis=0)
    
    return heatmap_fish
    
def test_transform(img, heatmap, transformation= lambda x:x, inverse= lambda x:x):
    
    print(img.shape)
    plt.imshow(array_to_img(img), interpolation="none")
    plt.show()
    
    img = transformation(img)
    
    print(img.shape)
    plt.imshow(array_to_img(img), interpolation="none")
    plt.show()
        
    heatmap_transformed = heatmap
    
    print(heatmap_transformed.shape)
    plt.imshow(heatmap_transformed, interpolation="none")
    plt.show()
        
    heatmap = inverse(heatmap_transformed)
    
    print(heatmap.shape)
    plt.imshow(heatmap, interpolation="none")
    plt.show()
    
    
def get_heatmap_from_transformation(img, model, transformation= lambda x:x, inverse= lambda x:x, debug=False, last=False, inception=False):
    
    if debug:
        print(img.shape)
        plt.imshow(array_to_img(img), interpolation="none")
        plt.show()
    
    img = transformation(img)
    
    if debug:
        print(img.shape)
        plt.imshow(array_to_img(img), interpolation="none")
        plt.show()
        
    heatmap_transformed = get_heatmaps(img, model, inception=inception)
    heatmap_transformed = scipy.misc.imresize(heatmap_transformed, (100,100), interp='bilinear', mode=None)/255
    if debug:
        plt.imshow(heatmap_transformed, interpolation="none")
        plt.show()
        
    heatmap = inverse(heatmap_transformed)
    if debug or last:
        plt.imshow(heatmap, interpolation="none")
        plt.show()
    return heatmap
    
    
# get paths for all images in the train forlder
def get_all_images(directory="train", regex = None):
    if regex is None:
        regex = './' + directory+'/*/*.jpg'
    a = list(glob.glob(regex))
    return [x.replace("\\", "/") for x in a]
    
    
def resolution(img_path):
    with Image.open(img_path) as im:
        return im.size
        
        
def add_to_dic_list(key, element, dic):
    if key in dic:
        dic[key].append(element)
    else:
        dic[key] = [element]
    return dic


        
def get_weights(mask,rects,i,Secu,x_r,y_r,opt=0):
    mask_size = mask.shape
    surf = mask_size[0]*mask_size[1]
    SUM = mask.sum()
    mask_weights = np.zeros(mask_size)
    act_surf = 0
    for rect in rects:
        x = rect['x']*x_r
        y = rect['y']*y_r
        w = rect['width']*x_r
        h = rect['height']*y_r
        if i == 0:
            act_surf += h*w
        if i == 1:
            act_surf += (h/2)*w
        if i == 2:
            act_surf += h*(w/2)
        if i == 3:
            act_surf += (h/2)*w 
        if i == 4:
            act_surf += h*(w/2)           
    surf = surf - act_surf
    for rect in rects:
        x = rect['x']*x_r
        y = rect['y']*y_r
        w = rect['width']*x_r
        h = rect['height']*y_r
        if opt == 0: # DIRAC
            mask_weights[np.where(mask == 1)] = surf/act_surf
            mask_weights[np.where(mask == 0)] = 1
            return mask_weights/mask_weights.max()
        elif opt == 1: # DIRAC
            mask_weights[np.where(mask == 1)] = surf/act_surf
            mask_weights[np.where(mask == 0)] = 1
            mask_weights = mask_weights/mask_weights.max()
            mask_weights = scipy.signal.convolve2d(mask_weights,np.ones((3,3))/9,mode='same')
            return mask_weights*(mask.shape[-1]*mask.shape[-2])/mask_weights.sum()
        

def get_5_masks_w(metadata, mask_size=(720,1280),secu=5,option=0):
    import math
    """ Get the mask of a given image. Save the results
        in the "mask" folder in case the function is 
        called again."""
    masks = np.zeros((5,mask_size[0],mask_size[1]))
    size = (metadata["height"], metadata["width"])
    try:
        rects = metadata['rectangles']
    except KeyError:
        if metadata["class"] == "NoF":
            rects = []
        else:
            return None
    if len(rects) > 0 :
        x_r = mask_size[1]/size[1]
        y_r = mask_size[0]/size[0]
        for rect in rects:
            x = rect['x']*x_r
            y = rect['y']*y_r
            w = rect['width']*x_r
            h = rect['height']*y_r
            X = math.floor(x)
            X2 = math.floor(x+w)
            Y = math.floor(y)
            Y2 = math.floor(y+h)
            masks[0,Y:Y2,X:X2] = 1
        for i in range(1,5):
            for rect in rects:
                x = rect['x']*x_r
                y = rect['y']*y_r
                w = rect['width']*x_r
                h = rect['height']*y_r
                if i == 1: # bottom
                    X = math.floor(x)
                    X2 = math.floor(x+w)
                    Y = math.floor(y)
                    Y2 = math.floor(y+h/2)
                    masks[i,Y:Y2,X:X2] = 1
                elif i == 2: # left
                    X = math.floor(x)
                    X2 = math.floor(x+w/2)
                    Y = math.floor(y)
                    Y2 = math.floor(y+h)                        
                    masks[i,Y:Y2,X:X2] = 1
                elif i  == 3: # top
                    X = math.floor(x)
                    X2 = math.floor(x+w)
                    Y = math.floor(y+h/2)
                    Y2 = math.floor(y+h)                           
                    masks[i,Y:Y2,X:X2] = 1
                elif i == 4: # right
                    X = math.floor(x+w/2)
                    X2 = math.floor(x+w)
                    Y = math.floor(y)
                    Y2 = math.floor(y+h)                           
                    masks[i,Y:Y2,X:X2] = 1
    
    masks = masks
    w_mask = get_my_weights(masks)
    return masks,w_mask
    
def get_my_weights(masks):

    mask = masks[0]
    inv_mask = 1-mask
    
    total_weights = mask.shape[0]*mask.shape[1]/2
    
    nb_ones = np.sum(mask)
    nb_zeros = np.sum(inv_mask)
    
    if nb_ones != 0:
        weights_ones = mask*total_weights/nb_ones
        weight_zeros = inv_mask*total_weights/nb_zeros
    else:
        weights_ones = mask
        weight_zeros = inv_mask*total_weights*2/nb_zeros
    
    return weight_zeros + weights_ones
    
def expand_mask_to_img(mask):
    mask = np.tile(mask,(3,1,1))
    return np.rollaxis(mask,0,3)
    
    
def plot_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    
    
def plot_5_masks(masks):
    for mask in masks.tolist():
        mask_img = expand_mask_to_img(np.array(mask))
        plot_img(mask_img) 