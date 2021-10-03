import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Input, ZeroPadding2D

def DarkConv(inputs, filters, layer_idx, stage_name, kernel_size=3, strides=(1, 1), padding="SAME"):
    layer_name = "{}_layer_{}".format(stage_name, str(layer_idx))
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False, name=layer_name+"_Conv")(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=layer_name+"_BN")(x)
    x = LeakyReLU(0.1, name=layer_name+"_Activation")(x)
    return x

def DarkPool(inputs, filters, layer_idx, stage_name):
    layer_name = "{}_layer_{}".format(stage_name, str(layer_idx))
    x = ZeroPadding2D(padding=((1,0), (0, 1)), name=layer_name+"_Pool")(inputs)
    x = DarkConv(x, filters, layer_idx, stage_name, strides=(2, 2), padding="VALID")
    return x 
    
def ResidualBlock(inputs, filters, layer_idx, stage_name):
    layer1, layer2 = layer_idx
    shortcut = inputs
    x = DarkConv(inputs,  filters, layer_idx=layer1, stage_name=stage_name, kernel_size=(1, 1))
    x = DarkConv(x     ,2*filters, layer_idx=layer2, stage_name=stage_name, kernel_size=(3, 3))
    x = shortcut + x
    return inputs

def Darknet53():
    inputs = Input(shape=(256, 256, 3))
    x = DarkConv(x, 32, 1, "stage0")
    x = DarkPool(x, 64, 2, "stage0")
    
    x = ResidualBlock(x, 32, [3, 4], "stage1")
    x = DarkPool(x, 128, 5, "stage1")
    
    x = ResidualBlock(x, 64, [6, 7], "stage2")
    x = ResidualBlock(x, 64, [8, 9], "stage2") 
    x = DarkPool(x, 256, 10, "stage2")
    
    x = ResidualBlock(x, 128, [11, 12], "stage3")
    x = ResidualBlock(x, 128, [13, 14], "stage3")
    x = ResidualBlock(x, 128, [15, 16], "stage3")
    x = ResidualBlock(x, 128, [17, 18], "stage3")
    x = ResidualBlock(x, 128, [19, 20], "stage3")
    x = ResidualBlock(x, 128, [21, 22], "stage3")
    x = ResidualBlock(x, 128, [23, 24], "stage3")
    x = ResidualBlock(x, 128, [25, 26], "stage3")
    x = DarkPool(x, 512, 27, "stage3")
    
    x = ResidualBlock(x, 256, [28, 29], "stage4")
    x = ResidualBlock(x, 256, [30, 31], "stage4")
    x = ResidualBlock(x, 256, [32, 33], "stage4")
    x = ResidualBlock(x, 256, [34, 35], "stage4")
    x = ResidualBlock(x, 256, [36, 37], "stage4")
    x = ResidualBlock(x, 256, [38, 39], "stage4")
    x = ResidualBlock(x, 256, [40, 41], "stage4")
    x = ResidualBlock(x, 256, [42, 43], "stage4")
    x = DarkPool(x, 1024, 44, "stage4")
    
    x = ResidualBlock(x, 512, [45, 46], "stage5")
    x = ResidualBlock(x, 512, [47, 48], "stage5")
    x = ResidualBlock(x, 512, [49, 50], "stage5")
    x = ResidualBlock(x, 512, [51, 52], "stage5")
    
    x = tf.keras.layers.GlobalAveragePooling2D(name="stage6_Average_Pooling")(x)
    x = tf.keras.layers.Dense(1000, activation=None, name="stage6_layer_53_Dense", use_bias=True)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs, x, name="Darknet53")
    return model