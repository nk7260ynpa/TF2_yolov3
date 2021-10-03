import tensorflow as tf

class img_cls_loader():
    def __init__(self, batch_size=64, img_size=224):
        self.batch_size = batch_size
        self.img_size = 224
        self.data_aug = True
        
    def load_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img)
        return img 
    
    def train_img_preprocess(self, img):
        img /= 255.
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return img 
    
    def valid_img_preprocess(self, img):
        img /= 255.
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return img 
    
    def tf_ds_gen(self, X, y, training=True, buffer_size=18000):
        AUTOTUNE = tf.data.AUTOTUNE
        X = tf.data.Dataset.from_tensor_slices(X)
        y = tf.data.Dataset.from_tensor_slices(y)
        X = X.map(self.load_img)
        if training == True:
            X.map(self.train_img_preprocess)
        else:
            X = X.map(self.valid_img_preprocess)
        data_ds = tf.data.Dataset.zip((X, y))
        data_ds = data_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        return data_ds
        
        