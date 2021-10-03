import os
import json
import pathlib2
import numpy as np
import pandas as pd
import tensorflow as tf


class Imagenet():
    def __init__(self, Imagenet_folder):
        self.data_folder = Imagenet_folder
        self.map_code = self.map_code_process() 
        self.X_train, self.y_train = self.train_data_process()
        self.X_valid, self.y_valid = self.valid_data_process()
        
    def map_code_process(self):
        map_code_path = os.path.join(self.data_folder, "imagenet_class_index.json")
        map_code = pd.read_json(map_code_path).T
        map_code.columns = ["Folder_code", "classes"]
        map_code["Model_code"] = np.arange(0, 1000)
        map_code = map_code[["Model_code", "Folder_code", "classes"]]
        return map_code
    
    def train_data_process(self):
        train_folder = os.path.join(self.data_folder, "train")
        train_data_root = pathlib2.Path(train_folder)
        map_dict = self.map_code.set_index("Folder_code")["Model_code"].to_dict()
        X_train, y_train = [], []
        Folder_Model_dict = self.map_code.set_index("Folder_code")["Model_code"].to_dict()
        for path in train_data_root.glob("./*/*.JPEG"):
            X_train.append(str(path))
            label_folder = path.parent.name
            model_code = Folder_Model_dict[label_folder]
            y_train.append(model_code)
        return X_train, y_train
    
    def valid_data_process(self):
        valid_folder = os.path.join(self.data_folder, "valid")
        valid_data_root = pathlib2.Path(valid_folder)
        X_valid = [str(path) for path in valid_data_root.glob("*.JPEG")]
        X_valid.sort()
        f = open(os.path.join(self.data_folder, "ILSVRC2012_validation_ground_truth.txt"), "r")
        valid_labels = f.readlines()
        f.close()
        y_valid = [int(label.rstrip("\n")) for label in valid_labels]
        return X_valid, y_valid


if __name__ == "__main__":
    #Test imagenet 
    dataset = Imagenet("datasets/Imagenet/2012")
    
    
    
    
    