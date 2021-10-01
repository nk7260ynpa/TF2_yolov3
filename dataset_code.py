import os
import json
import pathlib2
import numpy as np
import pandas as pd
import tensorflow as tf


class Imagenet_dataset(self):
    def __init__(self, Imagenet_folder):
        self.data_folder = Imagenet_folder
        self.map_code = self.map_code_process() 
        self.X_train, self.y_train = self.train_data_process()
        
    def map_process(self):
        map_code = pd.read_json(self.data_folder).T
        map_code.columns = ["imagenet_code", "classes"]
        map_code["Model_code"] = np.arange(0, 1000)
        map_code = map_code[["Model_code", "Folder_code", "classes"]]
        return map_code
    
    def train_data_process(self):
        train_data_root = pathlib2.Path(self.Imagenet_folder)
        map_dict = 
        X_train, y_train = [], []
        Folder_Model_dict = self.map_code.set_index("folder_code")["Model_code"].to_dict()
        for path in train_data_root.glob("./*/*.jpg"):
            X_train.append(str(path))
            label_folder = path.parents.name
            model_code = Folder_Model_dict[label_folder]
            y_train.append(model_code)
        return X_train, y_train
        
    def valid_data_process(self):
        
    
    
    
    