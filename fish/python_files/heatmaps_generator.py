import os
import glob
import matplotlib.pyplot as plt
from heatmap import to_heatmap
from heatmap import synset_to_dfs_ids
import matplotlib.image as mpimg
import numpy as np
from keras.applications.resnet50 import preprocess_input as preprocess_input_imagenet
import scipy
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, random_rotation, flip_axis, array_to_img



def get_heatmaps(img, model):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input_imagenet(x)

    out = model.predict(x)

    s = "n02512053" # Imagenet code for "fish"
    ids = synset_to_dfs_ids(s)
    heatmap_fish = out[0,ids].sum(axis=0)
    
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
    
    
    
def get_heatmap_from_transformation(img, model, transformation= lambda x:x, inverse= lambda x:x, debug=False, last=False):
    
    if debug:
        print(img.shape)
        plt.imshow(array_to_img(img), interpolation="none")
        plt.show()
    
    img = transformation(img)
    
    if debug:
        print(img.shape)
        plt.imshow(array_to_img(img), interpolation="none")
        plt.show()
        
    heatmap_transformed = get_heatmaps(img, model)
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
def get_all_images(directory="train/train", regex = None):
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


# Create a directory if it doesn't exist
def mk(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def get_mask_from_img(classe,img_name,size):
    import os
    import collections
    import json
    import numpy as np
    json_name = classe.lower()+'_labels.json'
    data_json = collections.defaultdict(int)
    mask = np.zeros((100,100))
    if json_name != 'nof_labels.json':
        with open(json_name,'r+') as f:
            labels = json.load(f)
            for label in labels:
                name = label['filename'].split('/')[-1]
                if img_name == name:
                    rects = label['annotations']
                    if len(rects) > 0 :
                        x_r = 100/size[0]
                        y_r = 100/size[1]
                        for rect in rects:
                            x = rect['x']*x_r
                            y = rect['y']*y_r
                            w = rect['width']*x_r
                            h = rect['height']*y_r
                            for n in range(size[0]-1):
                                for k in range(size[1]-1):
                                    if n > x and n < (x+w) and k > y and k < (y+h) and n < 100 and k < 100:
                                            mask[k][n] = 1
                       # img_array[idx]=mask[idx]
                        return mask
                    else:
                        return mask  
    else:
        return mask