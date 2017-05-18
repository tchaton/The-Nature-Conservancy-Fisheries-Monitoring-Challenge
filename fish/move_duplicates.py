import json
import os

# Create a directory if it doesn't exist
def mk(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        

TRAIN_FOLDER = "./train/"
JSON_PATH = "duplicates.json"

with open(JSON_PATH) as data_file:    
    duplicate_dict = json.load(data_file)
    
    
# We need to be able to find the path of a file:
path_dictionary = {}

for class_folder in os.listdir(TRAIN_FOLDER):
    files = os.listdir(TRAIN_FOLDER + class_folder)
    
    for file in files:
        path_dictionary[file] = TRAIN_FOLDER + class_folder + "/" + file
        

duplicate_folder = TRAIN_FOLDER + "duplicates/"
mk(duplicate_folder)

for file, duplicates in duplicate_dict.items():
    try:
        path = path_dictionary[file]
    except KeyError:
        print("Couldn't find", file, "in the train folder.")
        continue
    mk(duplicate_folder + path.split("/")[-2])    

    dest_folder = duplicate_folder + path.split("/")[-2] + "/" + file
    mk(dest_folder)

    for duplicate in duplicates:
        
        try:
            path = path_dictionary[duplicate]
        except KeyError:
            print("Couldn't find", file, "in the train folder.")
            continue
    
        new_path = dest_folder + "/" + duplicate

        os.rename(path, new_path)

