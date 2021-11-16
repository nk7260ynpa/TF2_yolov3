import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Softmax, Concatenate

#Model Module
# ===========================================================================================
def DarkConv(inputs, filters, kernel_size, layer_idx, name, strides=(1, 1), padding="SAME"):
    layer_name = "{}_layer_{}".format(name, str(layer_idx))
    inputs = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, 
                    name=layer_name+"_Conv")(inputs)
    inputs = BatchNormalization(momentum=0.9, epsilon=1e-05, name=layer_name+"_BN")(inputs)
    inputs = LeakyReLU(alpha=0.1, name=layer_name+"_Activation")(inputs)
    return inputs

def DarkPool(inputs, filters, layer_idx, name, strides=(2, 2), padding="VALID"):
    layer_name = "{}_layer_{}".format(name, str(layer_idx))
    inputs = ZeroPadding2D(((1, 0), (1, 0)), name=layer_name)(inputs)
    inputs = DarkConv(inputs, filters, (3, 3), layer_idx, name, strides, padding)
    return inputs

def ResidualBlock(inputs, filters, layer_idx, name, strides=(1, 1), padding="SAME"):
    layer1, layer2 = layer_idx
    shortcut = inputs
    inputs = DarkConv(inputs, filters   , (1, 1), layer1, name)
    inputs = DarkConv(inputs, filters*2 , (3, 3), layer2, name)
    inputs = shortcut + inputs
    return inputs

def upsample_conv(inputs, filters, output_shape, layer_idx,name):
    inputs = DarkConv(inputs, filters, kernel_size=1, layer_idx=layer_idx, name="upsample")
    inputs = tf.image.resize(inputs, (output_shape, output_shape), method="nearest")
    return inputs
# ===========================================================================================


#Darknet Model
# ===========================================================================================
def Darknet53(input_size=256, classes=1000):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3,), name="Input_stage")

    x = DarkConv(inputs, 32, (3, 3), 1, "stage0")
    x = DarkPool(x , 64,  2, "stage0")

    x = ResidualBlock(x, 32, [3, 4], "stage1")
    x = DarkPool(x , 128, 5, "stage1")

    x = ResidualBlock(x, 64, [6, 7], "stage2")
    x = ResidualBlock(x, 64, [8, 9], "stage2")
    x = DarkPool(x , 256, 10, "stage2")

    x = ResidualBlock(x, 128, [11, 12], "stage3")
    x = ResidualBlock(x, 128, [13, 14], "stage3")
    x = ResidualBlock(x, 128, [15, 16], "stage3")
    x = ResidualBlock(x, 128, [17, 18], "stage3")
    x = ResidualBlock(x, 128, [19, 20], "stage3")
    x = ResidualBlock(x, 128, [21, 22], "stage3")
    x = ResidualBlock(x, 128, [23, 24], "stage3")
    x = ResidualBlock(x, 128, [25, 26], "stage3")
    x = DarkPool(x , 512, 27, "stage3")

    x = ResidualBlock(x, 256, [28, 29], "stage4")
    x = ResidualBlock(x, 256, [30, 31], "stage4")
    x = ResidualBlock(x, 256, [32, 33], "stage4")
    x = ResidualBlock(x, 256, [34, 35], "stage4")
    x = ResidualBlock(x, 256, [36, 37], "stage4")
    x = ResidualBlock(x, 256, [38, 39], "stage4")
    x = ResidualBlock(x, 256, [40, 41], "stage4")
    x = ResidualBlock(x, 256, [42, 43], "stage4")
    x = DarkPool(x , 1024, 44, "stage4")

    x = ResidualBlock(x, 512, [45, 46], "stage5")
    x = ResidualBlock(x, 512, [47, 48], "stage5")
    x = ResidualBlock(x, 512, [49, 50], "stage5")
    x = ResidualBlock(x, 512, [51, 52], "stage5")

    x = GlobalAveragePooling2D(name="stage6_Average_Pooling")(x)

    x = Dense(classes, activation=None, name="stage6_layer_53_Dense", use_bias=True)(x)
    x = Softmax()(x)
    model = tf.keras.Model(inputs, x, name="Darknet")
    return model


def Darknet53_yolo_base(input_size=416):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3,), name="Input_stage")

    x = DarkConv(inputs, 32, (3, 3), 1, "stage0")
    x = DarkPool(x , 64,  2, "stage0")

    x = ResidualBlock(x, 32, [3, 4], "stage1")
    x = DarkPool(x , 128, 5, "stage1")

    x = ResidualBlock(x, 64, [6, 7], "stage2")
    x = ResidualBlock(x, 64, [8, 9], "stage2")
    x = DarkPool(x , 256, 10, "stage2")

    x = ResidualBlock(x, 128, [11, 12], "stage3")
    x = ResidualBlock(x, 128, [13, 14], "stage3")
    x = ResidualBlock(x, 128, [15, 16], "stage3")
    x = ResidualBlock(x, 128, [17, 18], "stage3")
    x = ResidualBlock(x, 128, [19, 20], "stage3")
    x = ResidualBlock(x, 128, [21, 22], "stage3")
    x = ResidualBlock(x, 128, [23, 24], "stage3")
    x = ResidualBlock(x, 128, [25, 26], "stage3")
    output_1 = x
    x = DarkPool(x , 512, 27, "stage3")

    x = ResidualBlock(x, 256, [28, 29], "stage4")
    x = ResidualBlock(x, 256, [30, 31], "stage4")
    x = ResidualBlock(x, 256, [32, 33], "stage4")
    x = ResidualBlock(x, 256, [34, 35], "stage4")
    x = ResidualBlock(x, 256, [36, 37], "stage4")
    x = ResidualBlock(x, 256, [38, 39], "stage4")
    x = ResidualBlock(x, 256, [40, 41], "stage4")
    x = ResidualBlock(x, 256, [42, 43], "stage4")
    output_2 = x
    x = DarkPool(x , 1024, 44, "stage4")

    x = ResidualBlock(x, 512, [45, 46], "stage5")
    x = ResidualBlock(x, 512, [47, 48], "stage5")
    x = ResidualBlock(x, 512, [49, 50], "stage5")
    x = ResidualBlock(x, 512, [51, 52], "stage5")
    output_3 = x

    model = tf.keras.Model(inputs, [output_1, output_2, output_3], name="Darknet_yolo")
    return model
    
def yolo_block(inputs, filters, name):
    inputs = DarkConv(inputs, filters  , kernel_size=1, layer_idx=1, name=name)
    inputs = DarkConv(inputs, filters*2, kernel_size=3, layer_idx=2, name=name)
    inputs = DarkConv(inputs, filters  , kernel_size=1, layer_idx=3, name=name)
    inputs = DarkConv(inputs, filters*2, kernel_size=3, layer_idx=4, name=name)
    inputs = DarkConv(inputs, filters  , kernel_size=1, layer_idx=5, name=name)
    route = inputs
    inputs = DarkConv(inputs, filters*2, kernel_size=3, layer_idx=6, name=name)
    return route, inputs

def Darknet_yolo(num_classes=80):
    starts_inputs = tf.keras.layers.Input(shape=(416, 416, 3))
    conv_base = Darknet53_yolo_base(input_size=416)
    
    #FPN route1
    conv_output_3 = tf.keras.layers.Input(shape=(13, 13, 1024), name="FPN_1_Input")
    upsample_output_1, inputs = yolo_block(conv_output_3, 512, name="FPN_1")
    detect_1 = Conv2D(filters=3*(5+num_classes), kernel_size=1, strides=1, name="FPN_1_output_Conv")(inputs)
    FPN_1_model = tf.keras.Model(inputs=conv_output_3, outputs=[upsample_output_1, detect_1], name="FPN_1")
    
    #FPN route2
    upsample_output_1 = tf.keras.layers.Input(shape=(13, 13 , 512), name="FPN_2_upsample_Input")
    inputs = upsample_conv(upsample_output_1, filters=256, output_shape=26, layer_idx=1, name="upsample")
    conv_output_2 = tf.keras.layers.Input(shape=(26, 26, 512), name="FPN_2_Input")
    inputs = Concatenate(axis=3)([inputs, conv_output_2])
    upsample_output_2, inputs = yolo_block(inputs, 256, name="FPN_2")
    detect_2 = Conv2D(filters=3*(5+num_classes), kernel_size=1, strides=1, name="FPN_2_output_Conv")(inputs)    
    FPN_2_model = tf.keras.Model(inputs=[upsample_output_1, conv_output_2], outputs=[upsample_output_2, detect_2], name="FPN_2")
    
    #FPN route3
    upsample_output_2 = tf.keras.layers.Input(shape=(26, 26 , 256), name="FPN_3_upsample_Input")
    inputs = upsample_conv(upsample_output_2, filters=128, output_shape=52, layer_idx=2, name="upsample")
    conv_output_1 = tf.keras.layers.Input(shape=(52, 52, 256), name="FPN_3_Input")
    inputs = Concatenate(axis=3)([inputs, conv_output_1])
    upsample_ouput_3, inputs = yolo_block(inputs, 128, name="FPN_3")
    detect_3 = Conv2D(filters=3*(5+num_classes), kernel_size=1, strides=1, name="FPN_3_output_Conv")(inputs)    
    FPN_3_model = tf.keras.Model(inputs=[upsample_output_2, conv_output_1], outputs=[detect_3], name="FPN_3")
    
    #Final output
    conv_output_1, conv_output_2, conv_output_3 = conv_base(starts_inputs)
    upsample_ouput_1, detect_1= FPN_1_model(conv_output_3)
    upsample_ouput_2, detect_2= FPN_2_model([upsample_ouput_1, conv_output_2])
    detect_3= FPN_3_model([upsample_ouput_2, conv_output_1])
    yolo_model = tf.keras.Model(starts_inputs, [detect_1, detect_2, detect_3])
    
    return yolo_model


def coco_pretrained_weights(weights_file, model):
    global ptr
    
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
    i = 0
    while i < len(model.variables) - 1:
        var1 = model.variables[i]
        var2 = model.variables[i+1]
        if "Conv" in var1.name:
            #print(var1.name.split('/')[-2])
            if "BN" in var2.name:
                gamma, beta, mean, var = model.variables[i+1: i+5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    var.assign(var_weights)
                i += 4   
                
            elif "Conv" in var2.name:
                #print(var2.name.split('/'))
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr+bias_params].reshape(bias_shape)
                ptr += bias_params
                bias.assign(bias_weights)
                i += 1    
            
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr+num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            var1.assign(var_weights)
            i += 1
            
            
def load_official_weights(weights_file, yolov3_model):
    global ptr
    ptr = 0
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("Darknet_yolo"))
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("FPN_1"))
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("FPN_2"))
    coco_pretrained_weights(weights_file, yolov3_model.get_layer("FPN_3"))
    
# ===========================================================================================
    
    