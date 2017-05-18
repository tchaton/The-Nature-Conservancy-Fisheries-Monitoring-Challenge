import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import pickle
import os
import h_gen
import threading


def to_categorical(label_s, nb_classes):
    """ One hot encode a vector, or a list of list of labels"""
    try:
        l = len(label_s)
        vector = np.zeros((l, nb_classes))
        for i in range(l):
            vector[i,int(label_s[i])]=1
        
    except TypeError:
        number = label_s
        vector = np.zeros(nb_classes)
        vector[number]=1
    return vector
    
def display_from_meta(dic):
    """ Given the metadata of an image, we plot everything we can concerning
        this image."""
    rectangles=[]
    points=[]
    first=False
    
    if "rectangles" in dic:
        rectangles=dic["rectangles"]
    if "head_tail" in dic:
        points = dic["head_tail"]
    if "ht_first" in dic:
        first = dic["ht_first"]
    display_image(dic["path"], rectangles, points, first)

def display_image(path, rectangles=[], points=[], first=False):
    im = np.array(Image.open(path), dtype=np.uint8)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    
    for i, rec in enumerate(rectangles):
        # Create a Rectangle patch
        rect = patches.Rectangle((rec["x"],rec["y"]),rec["width"],rec["height"],
                                linewidth=1,edgecolor='r',facecolor='none')
            
        if first and i==0:
            rect = patches.Rectangle((rec["x"],rec["y"]),rec["width"],rec["height"],
                                linewidth=1,edgecolor='b',facecolor='none')
            
        # Add the patch to the Axes
        ax.add_patch(rect)
        
    for point in points:
        # Create a Rectangle patch filled
        rect = patches.Rectangle((point["x"]-2,point["y"]-2),4,4,
                                linewidth=3,edgecolor='b',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
    
    
def save(variable, path):
    with open(path, "wb") as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)
        
def save_array(arr,fname):
    np.save(fname, arr)

        
def load(path):
    with open(path, "rb") as f:
        variable = pickle.load(f)
    return variable
    
def load_array(fname):
    return np.load(fname + ".npy")
    
    
# Create a directory if it doesn't exist
def mk(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    
    
def split_train_test(nb_images, train_dir):
    """ Use case : split_train_test(5,"boats_id_train/")"""
    temp = train_dir.find("train")
    test_dir = train_dir[:temp] + "test" + train_dir[temp+5:]
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    classes = os.listdir(train_dir)
        
    train_dir = [train_dir + x + "/" for x in classes]
    test_dir = [test_dir + x + "/" for x in classes]
    
    for folder in test_dir:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    for train_class, test_class in zip(train_dir, test_dir):
        files = os.listdir(train_class)[:nb_images]
        for file in files:
            copyfile(train_class + file,test_class + file)
            os.remove(train_class + file)
            
            
def display_mask(m, mask=None):
    im = np.array(Image.open(m["path"]), dtype=np.uint8)
    if mask is None:
        mask = h_gen.get_mask(m)
    im2 = Image.fromarray(mask*255)
    im2 = im2.resize((m["width"],m["height"]), Image.ANTIALIAS)
    plt.imshow(im)
    plt.imshow(im2, alpha=0.2)
    plt.show()
    
    
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
            
            
def apply_augmentations(batch, image_data_generator):
    for i in range(batch.shape[0]):
        batch[i] = image_data_generator.random_transform(batch[i])
        batch[i] = image_data_generator.standardize(batch[i])
    return batch
        
