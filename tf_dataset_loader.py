import os
import pathlib2
import numpy as np
import pandas as pd
import tensorflow as tf


class img_cls_loader():
    def __init__(self, batch_size=64, img_size=224):
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_aug = True
        
    def load_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        return img 
    
    def train_img_preprocess(self, img):
        img = tf.cast(img, tf.float32)
        img /= 255.
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return img 
    
    def valid_img_preprocess(self, img):
        img = tf.cast(img, tf.float32)
        img /= 255.
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return img 
    
    def tf_ds(self, X, y, training=True, buffer_size=10000):
        if training == True:
            random_index = np.arange(0, len(X))
            np.random.shuffle(random_index)
            X = np.array(X)[random_index]
            y = np.array(y)[random_index]
        AUTOTUNE = tf.data.AUTOTUNE
        X = tf.data.Dataset.from_tensor_slices(X)
        y = tf.data.Dataset.from_tensor_slices(y)
        X = X.map(self.load_img)
        if training == True:
            X = X.map(self.train_img_preprocess)
        else:
            X = X.map(self.valid_img_preprocess)
        data_ds = tf.data.Dataset.zip((X, y))
        data_ds = data_ds.shuffle(buffer_size=buffer_size).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        return data_ds

class Imagenet_ds():
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
        valid_path = os.path.join(self.data_folder, "map_clsloc.txt")
        Valid_map = pd.read_csv(valid_path, delimiter=" ", names=["Folder_code", "Valid_code", "classes"])
        map_code = pd.merge(map_code, Valid_map[["Folder_code", "Valid_code"]], on="Folder_code")
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
        Valid_map_dict = self.map_code.set_index("Valid_code")["Model_code"].to_dict()
        y_valid = [Valid_map_dict[int(label.rstrip("\n"))] for label in valid_labels]
        
        return X_valid, y_valid
        