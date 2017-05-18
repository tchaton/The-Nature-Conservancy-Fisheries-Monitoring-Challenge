import json
import glob
import h_gen
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tq
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from utils import *
import importlib
import h_gen
import utils

classes_dictionary = {"NoF":0, "OTHER":1, "ALB": 2, "BET":3, "DOL":4, "LAG":5, "SHARK":6, "YFT":7}
reverse_classes_dictionary = {"0":"NoF", "1":"OTHER", "2": "ALB", "3":"BET", "4":"DOL", "5":"LAG", "6":"SHARK", "7":"YFT"}

def get_classes_dictionary():
    return classes_dictionary

def get_reverse_classes_dictionary():
    return reverse_classes_dictionary

def generate_cluster_from_imgs(input_path,size=(200,200),alpha=0.8):
    import collections 
    import os
    from sklearn import cluster
    from sklearn import neighbors
    from scipy.misc import imread, imsave,imresize
    import pickle
    import numpy as np
    from tqdm import tqdm_notebook as tq
    import random
    #print('IT WILL SAVE DATA IN : data_clusters.p,cluster_0.p,cluster_1.p')
    classes = [f for f in os.listdir(input_path) if f!='.DS_Store']
    L = np.array([len(os.listdir(input_path+classe)) for classe in classes]).sum()
    knn_cls = 2
    channel = 3
    #nbr_bins = 255
    data = collections.defaultdict(int)
    data_0 = collections.defaultdict(int)
    data_1 = collections.defaultdict(int)
    data_0_0 = collections.defaultdict(int)
    data_1_1 = collections.defaultdict(int)   
    c = tq(total=L)
    for classe in classes:
        data[classe] = collections.defaultdict(int)
        data_0[classe] = []
        data_1[classe] = []
        data_0_0[classe] = []
        data_1_1[classe] = []        
        imgs = os.listdir(input_path+classe)
        training_imgs = len(imgs)
        imgs_path = [input_path+classe+'/'+img_name for img_name in imgs]
        training_files = np.array(sorted(imgs_path, key=lambda x: random.random()))
        training = np.array([imresize(imread(img),size, interp='bilinear', mode=None)  for img in training_files])
        training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
        training_features = np.zeros((training_imgs, 3))
        for i in range(training_imgs):
            c.update()
            training_features[i][0] = (training_means[i][0] - training_means[i][1])
            training_features[i][0] += (training_means[i][0] - training_means[i][2])
            training_features[i][1] = (training_means[i][1] - training_means[i][0])
            training_features[i][1] += (training_means[i][1] - training_means[i][2])
            training_features[i][2] = (training_means[i][2] - training_means[i][0])
            training_features[i][2] += (training_means[i][2] - training_means[i][1])

        kmeans = cluster.KMeans(n_clusters=knn_cls).fit(training_features)
        cluster_0 = training_files[np.where(kmeans.labels_ == 0)]
        cluster_1 = training_files[np.where(kmeans.labels_ == 1)]
        for i,img_name in enumerate(cluster_0):
            data[classe][img_name.split('/')[-1]]=0
            if int(alpha*len(cluster_0)) > i:
                data_0[classe].append(img_name.split('/')[-1])
            else:
                data_0_0[classe].append(img_name.split('/')[-1])

        for i,img_name in enumerate(cluster_1):
            data[classe][img_name.split('/')[-1]]=1
            if int(alpha*len(cluster_1)) > i:
                data_1[classe].append(img_name.split('/')[-1])
            else:
                data_1_1[classe].append(img_name.split('/')[-1])
    
    pickle.dump( data, open( "data_clusters.p", "wb" ) )
    pickle.dump( data_0, open( "cluster_0_train.p", "wb" ) )
    pickle.dump( data_1, open( "cluster_1_train.p", "wb" ) )
    pickle.dump( data_0_0, open( "cluster_0_test.p", "wb" ) )
    pickle.dump( data_1_1, open( "cluster_1_test.p", "wb" ) )    
    print("DONE")

def get_cluster_data(input_path):
    import pickle
    return pickle.load( open( input_path, "rb" ) )
    
def find_dic(f_name, list_of_dics):
    for dic in list_of_dics:
        file = dic["filename"].split("/")[-1]
        if file == f_name:
            return dic

def remove_class(list_of_dicts):
    new_list = []
    for dic in list_of_dicts:
        dic.pop("class")
        new_list.append(dic)
    return new_list
    

def is_in_rect(x,y,rect, margin=0):
    """ This function check if a point is in a rectangle, 
        the rectangle can be a dictionary or a tuple of 4 points"""
        
    m = margin
    if type(rect) == dict:
        ass1 = rect["x"] - m < x < rect["x"] + rect["width"] + m
        ass2 = rect["y"] - m < y < rect["y"] + rect["height"] + m
    else:
        ass1 = rect[0] - m < x < rect[2] + m
        ass2 = rect[1] - m < y < rect[3] + m
    return ass1 and ass2
    
def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return (range_overlap(r1["x"], r1["x"]+r1["width"], r2[0], r2[2]) and 
        range_overlap(r1["y"], r1["y"]+r1["height"], r2[1], r2[3]))
        
def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)
    
def head_tail_first(img_rect, ht):
    """ This function take a list of rectangles and put first 
        in list the rectangle that has the annotated head and tail."""
    x1 = ht[0]["x"]
    y1 = ht[0]["y"]
    x2 = ht[1]["x"]
    y2 = ht[1]["y"]
    index = None
    
    for i, rect in enumerate(img_rect):
        # We need to verify if the two points are in the rectangle. 
        # We add a margin m.
        m = 20
        
        ass1 = is_in_rect(x1, y1, rect, m)
        ass2 = is_in_rect(x2, y2, rect, m)
        if ass1 and ass2:
            index = i 
            
    # If the good rectangle was found
    if index is not None:
        
        # We put it first 
        rectangle_with_ht = img_rect.pop(index)
        img_rect.insert(0, rectangle_with_ht)
        return img_rect, True
    
    # If we didn't find the rectangle, we specify it as a second variable
    else:
        return img_rect, False 

def find_angle(ht):
    x = ht[0]["x"] - ht[1]["x"]
    y = ht[0]["y"] - ht[1]["y"]
    return np.angle(complex(x,y), True)
    

def create_metadata():
    
    utils.mk("mask/")
    paths = h_gen.get_all_images()
    jsons_r = h_gen.get_all_images(regex="../JSONS/*")
    print(jsons_r)
    jsons_ht = h_gen.get_all_images(regex="heads_tails/*")
    try:
        data_clusters = get_cluster_data("data_clusters.p")
    except:
        print("USE THE FUNCTION meta.generate_cluster_from_imgs(imgs_path)")
        return None
    # Loading the rectangles
    rectangles = []
    for json_path in jsons_r:
        with open(json_path,'r+') as f:
            labels = json.load(f)
            rectangles += labels
            
    # Loading the heads and tails
    heads_tails = []
    for json_path in jsons_ht:
        with open(json_path,'r+') as f:
            labels = json.load(f)
            heads_tails += labels
    
    # Fill the infos
    results = {}
    for path in tq(paths):

        dic = {}
        dic["path"] = path
        dic["filename"] = path.split("/")[-1]
        dic["class"] = path.split("/")[-2]
        dic["code"] = classes_dictionary[dic["class"]]
        size = h_gen.resolution(path)
        dic["height"] = size[1]
        dic["width"] = size[0]
        dic['cluster'] = data_clusters[dic["class"]][dic["filename"]]
        
        # The image doesn't necessarely has a fish
        ht = find_dic(dic["filename"], heads_tails)
        if ht is not None:
            ht = remove_class(ht["annotations"])
            if len(ht) == 2:
                dic["head_tail"] = ht
                dic["angle"] = find_angle(ht)
            else:
                ht=None

        img_rect = find_dic(dic["filename"],rectangles)
        if img_rect is not None:
            img_rect = remove_class(img_rect["annotations"])
            if ht is not None:
                img_rect, found = head_tail_first(img_rect, ht)
                dic["ht_first"] = found
            dic["rectangles"] = img_rect
        results[dic["filename"]] = dic
        
        
    return results
    

def create_metadata_test(sub_fold):
    paths = h_gen.get_all_images(regex="./TestT/*.jpg")
    filenames = [x.split("/")[-1] for x in paths]
    heights = [h_gen.resolution(x)[1] for x in paths]
    widths = [h_gen.resolution(x)[0] for x in paths]
    
    return list(zip(paths, filenames,heights,widths))
    
    


