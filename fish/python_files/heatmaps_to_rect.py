import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import pickle
import os
import h_gen
from numpy import unravel_index
import itertools
from keras.layers import *
from tqdm import tqdm_notebook as tq
from keras.models import Model


def create_kernel(border_size, width, height, kernel_size):
    
    # The full part
    kernel = np.zeros((5, kernel_size, kernel_size))
    
    x_middle = int(kernel.shape[2]/2)
    y_middle = int(kernel.shape[1]/2)
    x_pad = int((kernel_size - width - 2*border_size)/2)
    y_pad = int((kernel_size - height - 2*border_size)/2)
    x_pad2 = x_pad + border_size
    y_pad2 = y_pad + border_size

    
    kernel[0,y_pad: -y_pad,x_pad: -x_pad] = -1
    kernel[0, y_pad2:-y_pad2, x_pad2: -x_pad2] = 1

    # We need to remember that theano flip the kernels before applying it.
    # The upper part:
    kernel[1, y_pad2: y_middle, x_pad2: -x_pad2] = -1
    kernel[1, y_middle: -y_pad2, x_pad2: -x_pad2] = 1
    
    # The left part:
    kernel[2, y_pad2: -y_pad2, x_pad2: x_middle] = -1
    kernel[2, y_pad2: -y_pad2, x_middle: -x_pad2] = 1
    
    # The lower part:
    kernel[3] = -kernel[1]
    
    # The right part:
    kernel[4] = -kernel[2]
    
    # Now we normalize by the size of our window
    factor = 4 * height * width + (border_size*2 + height) * (border_size*2 + width)
    
    kernel = kernel*1000/factor
    
    return kernel 
    
    
def get_best_box(matrix, params, threashold=0):
    max_position = unravel_index(matrix.argmax(), matrix.shape)
    if np.max(matrix) > threashold:
        best_params = params[max_position[0]]
        xi =  int(max_position[2] - (best_params[1]/2))
        yi =  int(max_position[1] - (best_params[2]/2))
        return yi, xi, best_params[2]+1, best_params[1]+2
    else:
        return None

        
def compute_border(x,y, bconf):
    surface = x*y
    
    # There is a linear relation between the surface and
    # the border size. (border = a*surface+b)
    x1 = bconf[0][0]**2
    x2 = bconf[1][0]**2
    
    a = (bconf[1][1]-bconf[0][1])/(x2-x1)
    b = bconf[0][1] - a*x1
    
    
    return int(a*surface+b)


def find_rectangles(heatmaps, threshold=300, ranges=(11,71), clip=0.20, debug=False, debug2=False,
                    border_conf=[(9,3),(71,10)], batch_size=1, max_fish=10, scores=False):
    """ heatmaps: The heatmaps. Must be of shape (big_batch_size, 5, height, width)
        threashold: Minimum score of a rectangle
        ranges: the minimum and maximum sizes of a rectangle in pixels
        clip: When a rectangle is found on a heatmap, all the values inside the rectangle are clipped
        border_conf: The size of the border to apply. [(9,3),(71,10)] means that a rectangle of size
                     9*9 should have a border of size 3 and a rectangle of size 71*71 should have a border of 
                     size 10. Everything in between is scaled with the surface of the rectangle.
        batch_size: Since the convolutions are made on GPU, there is a batch size to assure memory requirements.
                    Higher values can make the algorithm go faster can create a CUDA memory error."""
    
    
    # This is because the clip operation is done on "heatmaps" and 
    # it would be sad if the function were to modify an argument.
    heatmaps = np.copy(heatmaps)
    
    kernel_size = ranges[1] + 2*compute_border(ranges[1], ranges[1], border_conf)

    
    params_to_test=[(compute_border(x,y, border_conf), y, x) \
                    for y,x in itertools.product(list(range(ranges[0],ranges[1],2)), repeat=2)]

    l = len(params_to_test)

    if debug:
        print("There are", l, "rectangles configurations")
        print("The kernel size is", kernel_size)
    
    kernels = np.zeros((l, 5, kernel_size, kernel_size))

    for i, (a,b,c) in enumerate(params_to_test):
        kernels[i] = create_kernel(a,b,c, kernel_size) 
        
    
    # Let's try to run the convolution on the gpu:
    img_a = Input(shape=(5, None, None))
    x = Convolution2D(l, kernel_size, kernel_size, bias=False, weights=[kernels], border_mode='same')(img_a)
    model = Model(img_a, x)

    if debug2:
        for i in range(5):
            print("min", np.min(kernels[10,i]))
            print("max", np.max(kernels[10,i]))
            print("border", params_to_test[10][0])
            plt.imshow(kernels[10,i])
            plt.show()
    
    
    rectangles = []
    scores = []
    
    # Now we loop over the images:
    for i in tq(range(len(heatmaps))):
        batch = heatmaps[i:i+1]
        
        rectangles_batch = [] 
        scores_batch = []
        # Contains the rectangles of image i.
        # The format is y, x, height, width
        
        for _ in range(max_fish):
            
            result = model.predict(batch)
            rec = get_best_box(result[0], params_to_test)
            if debug:
                
                plt.imshow(batch[0,0])
                plt.show()
                plt.imshow(result[0][0])
                plt.show()
                print("max found at position " , unravel_index(result.argmax(), result.shape))
                print("max " , np.max(result))
                print("min " , np.min(result))
                print(rec)

                fig,ax = plt.subplots(1)
                ax.imshow(batch[0,0])
                rect = patches.Rectangle((rec[1],rec[0]),rec[3],rec[2],
                                                linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                plt.show()
                
            # This is our cue to stop digging for rectangles
            if np.max(result[0]) < threshold:
                break
            
            rectangles_batch.append(rec)
            scores_batch.append(np.max(result[0]))
            
            pointer_on_masks = batch[0,:,rec[0]: rec[0]+rec[2], rec[1]: rec[1]+rec[3]]

            np.clip(pointer_on_masks, -float("inf"), 0.1, pointer_on_masks)
        rectangles.append(rectangles_batch)
        scores.append(scores_batch)
        
    
    if scores:
        return rectangles, scores
    
    return rectangles


def create_model_rect(threshold=300, ranges=(11,71), border_conf=[(9,3),(71,10)]):
    kernel_size = ranges[1] + 2*compute_border(ranges[1], ranges[1], border_conf)   
    params_to_test=[(compute_border(x,y, border_conf), y, x) \
                    for y,x in itertools.product(list(range(ranges[0],ranges[1],2)), repeat=2)]

    l = len(params_to_test)

    kernels = np.zeros((l, 5, kernel_size, kernel_size))

    for i, (a,b,c) in enumerate(params_to_test):
        kernels[i] = create_kernel(a,b,c, kernel_size) 


    # Let's try to run the convolution on the gpu:
    img_a = Input(shape=(5, None, None))
    x = Convolution2D(l, kernel_size, kernel_size, bias=False, weights=[kernels], border_mode='same')(img_a)
    model = Model(img_a, x)
    return model, params_to_test




















