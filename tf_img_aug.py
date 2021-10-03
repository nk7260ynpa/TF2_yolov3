import tensorflow as tf
import tensorflow_addons as tfa

def img_random_rotate(img, degree):
    degree = tf.random.uniform([], minval=-1*degree, maxval=degree)
    img = tfa.image.rotate(img, degree*(2*np.pi/360.0))
    return img

def img_random_RGB(img):
    rand_num = tf.random.shuffle(tf.range(3))
    R = img[..., rand_num[0]]
    G = img[..., rand_num[1]]
    B = img[..., rand_num[2]]
    img = tf.stack([R, G, B], axis=2)
    return img

def img_random_blur(img, blur_rate):
    random_matrix = tf.random.uniform((img[0], img[1], 1), minval=0, maxval=1)
    random_matrix = tf.cast(random_matrix > (1-blur_rate), tf.float32)
    img = img * random_matrix
    return img

def img_standardize(img):
    img -= 0.5
    img /= 0.5
    return img

