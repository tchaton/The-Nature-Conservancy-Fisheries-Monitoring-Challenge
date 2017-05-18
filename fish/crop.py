import numpy as np
from PIL import Image
from tqdm import tqdm_notebook as tq

import h_gen
import meta


# this generates the rectangles to crop.
# size is the average size of rectangles to crop
# there is also a variation in the size that can be tuned
def rectangle_generator(height, width, size=150, variation=0.3, rectangles=[]):
    xs = np.linspace(0, width, 8)
    ys = np.linspace(0, height, 8)
    for x in xs[1:-1]:
        for y in ys[1:-1]:
            sides = np.random.uniform(size*(1-variation), size*(1+variation), (2,))
            #print("x=" + str(x) + "   y=" + str(y))
            rect = (int(x-sides[0]/2), int(y-sides[1]/2), int(x+sides[0]/2), int(y+sides[1]/2))
            overlapped = False
            for rectangle in rectangles:
                overlapped = meta.overlap(rectangle, rect)
                if overlapped:
                    break
            if not overlapped:
                yield rect

                
                
def crop_images(metadata):
    """ This function crop the fishes as well as part of boats and put everything 
        in the corresponding folder."""
    # We create the folders
    h_gen.mk("cropped")
    for key, value in meta.classes_dictionary.items():
        h_gen.mk("cropped/" + key)

    # We crop the images
    for key, value in tq(metadata.items()):
        
        img = Image.open(value["path"])
        rectangles = []
        # First, we crop the fishes
        if "rectangles" in value:
            rectangles = value["rectangles"]
            for i, rect in enumerate(rectangles):
                path_to_save = "cropped/" + value["class"] + "/" + value["filename"][:-4] + "_" + str(i) + value["filename"][-4:]
                
                # We want the fishes to be in squares, so we need to find the center
                side = max(rect["width"], rect["height"])
                x = rect["x"] + rect["width"]/2
                y = rect["y"] + rect["height"]/2
                m = 15
                
                cropped = img.crop((x-side/2-m , y-side/2-m, x+side/2+m , y+side/2+m))
                cropped.save(path_to_save)
        
        # Now we do crop of the background
        for i, rect in enumerate(rectangle_generator(value["width"], value["height"], 230, 0.3, rectangles)):
            cropped = img.crop(rect)
            path_to_save = "cropped/NoF/" + value["filename"][:-4] + "_" + str(i) + value["filename"][-4:]
            cropped.save(path_to_save)
            
