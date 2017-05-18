from keras.optimizers import SGD
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tq
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.utils.layer_utils import layer_from_config
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import itertools
import gc
from keras.preprocessing import image
from keras.models import model_from_json
from utils import *


def to_str(img_size):
    return str(img_size[0])+'_'+str(img_size[1])+'/'

def generate_intermediary_vectors(pictures_meta, pictures_sizes, fold_precompute, path_model_precompute):

    model = None
    for img_size in pictures_sizes:
        dim = to_str(img_size)
        folder = fold_precompute + dim
        mk(folder)
        f = os.listdir(folder)
        X = np.zeros((1, 3)+img_size)
        
        for path, f_name, height, width in tq(pictures_meta):
            if f_name + ".npy" not in f:
                
                if model is None:
                    model = load_model(path_model_precompute)
                
                X[0] = img_to_array(load_img(path, target_size=img_size))
                X = preprocess_input(X)
                y = model.predict(X)[0]
                save_array(y, folder + f_name)
                

def create_heatmaps(pictures_meta, fold_precompute, pictures_sizes, path_model_heatmaps, fold_heatmaps):
    
    heatmaps_batch = None
    model = None
    
    folders = [fold_precompute + to_str(img_size) for img_size in pictures_sizes]
    f_list = os.listdir(fold_heatmaps)
    for i, (path, f_name, height, width) in enumerate(tq(pictures_meta)): 
        file = fold_heatmaps + f_name
        
        if f_name + ".npy" not in f_list:
            
            if model is None:
                model = load_model(path_model_heatmaps)
            
            paths = [x + f_name for x in folders]
            vectors = [load_array(x) for x in paths]
            vectors = [np.expand_dims(x,axis=0) for x in vectors]
            heatmaps = model.predict(vectors)
            heatmaps = np.array(heatmaps)
            heatmaps = np.swapaxes(heatmaps,0,1)
            heatmaps = np.reshape(heatmaps, (5,) + vectors[-1][0][0].shape)
            save_array(heatmaps, file)
        else:
            heatmaps = load_array(file)
        
        if heatmaps_batch is None:
            heatmaps_batch = np.zeros((len(pictures_meta),) + heatmaps.shape)
        
        heatmaps_batch[i] = heatmaps

    return heatmaps_batch
        
        
def show_heatmaps(heatmaps, pictures_meta, index, mask_idx=0, nb_to_show = 10):

    for i in range(index,index+nb_to_show):

        h = heatmaps[i, mask_idx]
        print(np.max(h))
        plt.figure(figsize=(24,24))
        plt.subplot(1,2,1)
        plt.imshow(np.array(Image.open(pictures_meta[i][0]), dtype=np.uint8))
        plt.subplot(1,2,2)

        plt.imshow(h)
        plt.show()   
        
        
def heatmaps_png(heatmaps, pictures_meta, folder):
    mk(folder)
    
    for h5, (path, f_name, height, width) in zip(heatmaps,pictures_meta):
        for i, h in enumerate(h5):
            plt.imshow(h)
            plt.savefig(folder + f_name + "_" + str(i) + ".jpg")
            

def detect_boats(pictures_names, boat_classifier_file, treashold):
    pass

    
def resize_rectangles(rectangles, pictures_meta, h_width, h_height):
    resized_rectangles = []
    
    for rects, (path, f_name, height, width) in zip(rectangles,pictures_meta):
        img_rectangles = []
        
        for rect in rects:
            x = rect[1] * width/ h_width
            y = rect[0] * height/ h_height
            rect_width = rect[3] * width/ h_width
            rect_height = rect[2] * height/ h_height
            img_rectangles.append((y,x,rect_height, rect_width))
        resized_rectangles.append(img_rectangles)
    return resized_rectangles
    
    
def print_rectangles(new_rectangles, scores, pictures_meta, index, nb_to_show = 10):
    
    for i in range(index,index+nb_to_show):
        fig,ax = plt.subplots(1)
        ax.imshow(np.array(Image.open(pictures_meta[i][0]), dtype=np.uint8))
        
        for rec, score in zip(new_rectangles[i], scores[i]):
            
            ax.text(rec[1],rec[0], str(round(score,3)), fontsize=15, color = "r")
            rect = patches.Rectangle((rec[1],rec[0]),rec[3],rec[2],
                                linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            
        plt.show()
        
def crop_images(pictures_meta, rectangles):
    pass

def species_classifier(cropped, path_model_predictions):
    pass

def combine_probabilities(p_species, scores):
    pass

def to_csv(final_predictions):
    pass


