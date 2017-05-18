from keras.preprocessing.image import load_img, img_to_array, array_to_img, Iterator
import numpy as np
from keras.applications.imagenet_utils import preprocess_input as preprocess_imagenet
from keras.applications.inception_v3 import preprocess_input as preprocess_inception



def resize_array(array, size, nearest_neighbor = False):
    """ Resize a numpy array. 
        Applies directly on a numpy array. 
        Not on a PIL image.
    """
    
    if nearest_neighbor:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
    
    a = array.transpose(1,2,0)
    a = cv2.resize(a, size, interpolation)
    a = a.transpose(2,0,1)
    return a
    
def adapt_list(data):
    if len(data) == 1:
        return data[0]
    else:
        return data

        
def adapt_tuple(X,Y,W=None):
    X = adapt_list(X)
    Y = adapt_list(Y)
    
    if W is not None:
        W = adapt_list(W)
        return X, Y, W
    else:
        return X, Y
        
def expand_until_match(base_array, array_to_match):
    nb_expand_dims = len(array_to_match.shape) - len(base_array.shape)
    result = base_array
    for _ in range(nb_expand_dims)
        result = np.expand_dims(result, 20)
    return result
    
    
def show_generator_results(generator, model=None, shuffle=None):
    """ Function to see what the generator generates. 
        Also allow to see what the model predicts if a model is provided.
        Shuffle should be used like this: shuffle=4
        if you want a batch at random among the first 4 batches
        that the generator generates.
    """
   
    X = None
    Y = None
    W = None
    Z = None
    
    if shuffle is None:
        batch = 0
    else:
        batch = int(np.random.uniform(0, shuffle))
    
    
    i = 0
    for dummy_variable in generator:
        if i == batch:
            break
        i+=1
        
    
    if len(dummy_variable) == 2:
        X, Y = dummy_variable
    elif len(dummy_variable) == 3:
        X, Y, W = dummy_variable
    else:
        print("Error concerning the output of the generator.")
        print("Expected 2 or 3 variable, had", len(dummy_variable))
        assert False
        
        
    if model is not None:
        Z = model.predict(X)
               
        
    # Now we'll convert the shape of everything so that it's 
    # easier to print everything.
    # Every variable will be in a list. As if there were always
    # multiple inputs and outputs.

    mandatory_values = (X, Y)
    
    for i in range(2):
        if type(mandatory_values[i]) == np.ndarray:
            mandatory_values[i] = [mandatory_values[i]]
    
    X, Y = mandatory_values # I'm not sure it does anything. But just in case.
    
    
    batch_size = X[0].shape[0]
    
    optional_values = (W, Z)
    
    for i in range(2):
        if optional_values[i] is None:
            optional_values[i] =[[None for _ in range(batch_size)]]
            
        if type(optional_values[i]) == np.ndarray:
            optional_values[i] = [optional_values[i]]
            
    
    for i in tange(batch_size):
        x = [a[i] for a in X]
        y = [a[i] for a in Y]
        w = [a[i] for a in W]
        z = [a[i] for a in Z]
    

def random_transform_mask(generator, x, y):
        # Need to modify to transform both X and Y ---- to do
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = generator.row_index - 1
        img_col_index = generator.col_index - 1
        img_channel_index = generator.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if generator.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-generator.rotation_range, generator.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if generator.height_shift_range:
            tx = np.random.uniform(-generator.height_shift_range, generator.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if generator.width_shift_range:
            ty = np.random.uniform(-generator.width_shift_range, generator.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if generator.shear_range:
            shear = np.random.uniform(-generator.shear_range, generator.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if generator.zoom_range[0] == 1 and generator.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(generator.zoom_range[0], generator.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=generator.fill_mode, cval=generator.cval)
        # For y, mask data, fill mode constant, cval = 0
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode="constant", cval=0)

        if generator.channel_shift_range != 0:
            x = random_channel_shift(x, generator.channel_shift_range, img_channel_index)

        if generator.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if generator.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y


def create_zones(dictionary, target_mask_size):
    """ Create a simple mask with zones (no values, they come later).
        The idea is to use this to create the final mask
        or the mask of weights for example.
        
        no fish zones should always be 0, fish zones always be 1. 
        Other zones should takes values like 2,3,4 for consistency.
    """
    pass

# We'll try to buid here an extra modulable custom keras generator.
# The goal is to be able to try to generate many things without having to
# change the code of the genrator.

class CustomIterator(Iterator):
    """ We'll try to buid here an extra modulable custom keras generator.
        The goal is to be able to try to generate many things without having to
        change the code of the genrator.
    """

    def __init__(self, metadata, target_size, target_mask_size=None, f_create_zones=None, 
                 values_mask=None, values_weights =None, preprocess = "imagenet", files=None, 
                 augmentation_generator=None, sample_weights=None, batch_size=32, shuffle=True, seed=None):
        """ Size are (width, height)."""
    
        # First, we need a list of files to use and iterate on:
        if files is None:
            files = [key for key, value in metadata.items()]
        
        # If the user didn't specify a function to create zones, the default one is used
        if f_create_zones is None:
            f_create_zones = create_zones
            
        # To put values in values in the mask, we can either have a function
        # or a tuple.
        if type(values_mask) == tuple:
            def values_mask(zone_matrix):
             
        
        # If target_size is a single value, we must put it in a list 
        # for consistency with the rest of the code
        
        if type(target_size[0]) != tuple:
            target_size = [target_size]
        
        if type(target_mask_size[0]) != tuple:
            target_mask_size = [target_mask_size]
            
        if target_mask_size is not None and values_mask is not None:
            print("You've given a target_mask_size but not")
            print("values to fill in the mask.")
            assert False
                
                
        # We make the preprocessing function:
        if preprocess == "imagenet":
            self.preprocess = preprocess_imagenet
        elif preprocess == "inception":
            self.preprocess = preprocess_inception
        elif type(preprocess) = function:
            self.preprocess = preprocess
        elif preprocess is False:
            self.preprocess = lambda x: x
        else:
            print("Wrong preprocess argument: ", preprocess)
            assert False
            
        if sample_weights is not None:
            if type(sample_weights) == dict:
                sample_weights = np.array([sample_weights[file] for file in files])
            elif type(sample_weights) == list:
                sample_weights = np.array(sample_weights)
                
        if type(values_mask) not in [tuple, function]:
            print("Wrong values_mask type, expected a tuple or a function.")
            print("Instead got ", type(values_mask))
            assert False
            
        
        self.metadata = metadata
        self.target_size = target_size
        self.target_mask_size = target_mask_size
        self.f_create_zones = f_create_zones
        self.values_mask = values_mask
        self.values_weights = values_weights
        self.augmentation_generator = augmentation_generator
        self.files = files
        
        
        super(DirectoryIterator, self).__init__(len(files), batch_size, shuffle, seed)

    def next(self):
        
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        
        
        # We'll create the placeholders:
        X = [np.zeros((current_batch_size, 3) + size) in self.target_size]
        categories = np.zeros((current_batch_size, 8))
        
        # Will be initialized later if needed
        masks = None
        masks_weights = None
        
        for j, i in enumerate(index_array):
        
            meta = metadata[self.files(i)]
            
            # Let's load the image.
            base_img = load_img(meta["path"])
            img_array = img_to_array(base_img)
            
            # Now we set the label.
            categories[j, meta["code"]] = 1
            
            # Now we do the masks
            if self.target_mask_size is not None:
                base_zone = self.f_create_zones(meta)
                
                if len(base_zone.shape) == 0:
                
                
                if self.augmentation_generator is not None:
                    img_array, base_zone = random_transform_mask(self.augmentation_generator, img_array, base_zone)
                    img_array = self.augmentation_generator.standardize(img_array)
                
                # We expand the mask
                zones = [resize_array(base_zone, size, nearest_neighbor=True) for size in self.target_mask_size]
                
                
                # We need to make the masks matrix if they're not already made
                # They don't have the right shape, but it'll be changed later.
                if masks is None:
                    masks = [np.zeros((current_batch_size,) + matrix.shape) for matrix in zones]
                
                
                # Based on the zones, we fill the masks.
                current_masks = [m[j] for m in masks]
                for zone_matrix, mask in zip(zones, current_masks):
                
                    # If values_mask is a tuple:
                    if type(self.values_mask) == tuple:
                        for i, value in enumerate(self.values_mask):
                            mask[zone_matrix==i] = value
                    
                    # If this is a function:
                    else:
                        self.values_mask(zone_matrix, mask)
                            
                            
                # Based on the zones, we fill the masks weights:
                if self.values_weights is not None:
                    
                    if masks_weights is None:
                        masks_weights = [np.zeros((current_batch_size,) + matrix.shape) for matrix in zones]
                    
                    
                    # Based on the zones, we fill the masks.
                    current_masks_weights = [m[j] for m in masks_weights]
                    for zone_matrix, mask_weights in zip(zones, current_masks_weights):
                    
                        # If values_mask is a tuple:
                        if type(self.values_mask) == tuple:
                            for i, value in enumerate(self.values_weights):
                                mask_weights[zone_matrix==i] = value
                        
                        # If this is a function:
                        else:
                            self.values_weights(zone_matrix, mask_weights)
                          
                            
                
            else:
                if self.augmentation_generator is not None:
                    img_array = self.augmentation_generator.random_transform(img_array)
                    img_array = self.augmentation_generator.standardize(img_array)
            
            
            # We put the images in the placeholders
            for k, size in enumerate(self.target_size):
                X[k][j] = resize_array(img_array, size)
            
            
              
                
        

        X = [self.preprocess(img_array) for img_array in X]
        
        # We concatenate the outputs:
        # The order should be [categories, masks]
        Y = [categories]
        if self.target_mask_size is not None:
            
            for mask_array in masks:
                for i in range(mask_array.shape[1]):
                    Y.append(mask_array[:,i,:,:])
                
        
        
        # We concatenate the weights:
        # The order should be [categories_weights, masks_weights]
        
        if self.values_mask is not None or self.sample_weights is not None:
            W = [np.ones((current_batch_size,))]
            
            if self.values_mask is not None:
                for mask_array_weights in masks_weights:
                    for i in range(mask_array_weights.shape[1]):
                        W.append(mask_array_weights[:,i,:,:])
                        
            if self.sample_weights is not None:
                batch_weights = self.sample_weights[index_array]
                for i in range(len(W)):
                    
                
            
            
            return adapt_tuple(X, Y, W)
        else:
            return adapt_tuple(X, Y)
        
        