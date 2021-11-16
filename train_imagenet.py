import time
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils import gpugrowth
import argparse
from tf_dataset_loader import Imagenet_ds, img_cls_loader
from yolov3_models import Darknet53
from utils import gpugrowth



parser = argparse.ArgumentParser()
parser.add_argument("--Imagenet_path", type=str, default="datasets/Imagenet/2012", help="Where Imagenet dataset")
parser.add_argument("--img_size", type=int, default=128, help="Image load size")
parser.add_argument("--batch_size", type=int, default=256, help="Training batch_size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=30, help="Training Epochs")
parser.add_argument("--save_weights", type=str, default="weights/darknet.h5", help="Weights save path")
parser.add_argument("--gpu_num", type=str, default='0', help="GPU number")
parser.add_argument("--random_seed", type=int, default=1048596, help="El psy congroo")


opt = parser.parse_args()

IMAGENET_PATH = opt.Imagenet_path
IMG_SIZE = opt.img_size
BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
GPU_NUM = opt.gpu_num
SAVE_WEIGHTS_PATH = opt.save_weights
RANDOM_SEED = opt.random_seed
LR = opt.lr

gpugrowth(GPU_NUM)

#Setting Random Seed
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#Load Imagenet path and lobel in numpy type
Imagenet = Imagenet_ds(IMAGENET_PATH)
X_train, y_train = Imagenet.X_train, Imagenet.y_train
X_valid, y_valid = Imagenet.X_valid, Imagenet.y_valid

#Load Imagent to tf type
data_loader = img_cls_loader(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
train_ds = data_loader.tf_ds(X_train, y_train, training=True)
valid_ds = data_loader.tf_ds(X_valid, y_valid, training=False)

#Model
model = Darknet53(IMG_SIZE, 1000)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
radam = tfa.optimizers.RectifiedAdam(learning_rate=LR)
optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

#Metrics
train_loss = tf.keras.metrics.Mean(name="Train_loss")
train_acc  = tf.keras.metrics.SparseCategoricalAccuracy(name="Train_acc")
valid_loss = tf.keras.metrics.Mean(name="Valis_loss")
valid_acc  = tf.keras.metrics.SparseCategoricalAccuracy(name="Valid_acc")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_acc(labels, predictions)

@tf.function
def valid_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)
    valid_loss(loss)
    valid_acc(labels, predictions)

    
_, __ = next(iter(train_ds))
_ = model(_)

best_loss = np.inf
print("=========== Training Start ===========")
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    valid_loss.reset_states()
    valid_acc.reset_states()
    
    train_start_time = time.time()
    for images, labels in train_ds:
        train_step(images, labels)
        train_end_time = time.time() - train_start_time
        print("Epoch {}, Train loss: {:.3f}, Train Acc: {:.3f}, {} seconds".format(
             epoch+1, train_loss.result().numpy(), train_acc.result().numpy(), int(train_end_time)), end='\r')  
    print()
    
    valid_start_time = time.time()
    for valid_images, valid_labels in valid_ds:
        valid_step(valid_images, valid_labels)
        valid_end_time = time.time() - valid_start_time
        print("Epoch {}, Valid loss: {:.3f} Valid Acc: {:.3f}, {} seconds".format(
             epoch+1, valid_loss.result().numpy(), valid_acc.result().numpy(), int(valid_end_time)), end='\r')
    print()
    
    template = ("Epoch {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Valid Loss: {:.3f}, Valid Acc: {:.3f}")
    
    print(template.format(epoch+1, train_loss.result(), train_acc.result(),
                          valid_loss.result(), valid_acc.result()))
    if valid_loss.result() < best_loss:
        model.save_weights(SAVE_WEIGHTS_PATH)
        best_loss = valid_loss.result()
        print("Save weights with loss:{:.3f}".format(best_loss.numpy()))
    print()
print("=========== Training End ===========")

