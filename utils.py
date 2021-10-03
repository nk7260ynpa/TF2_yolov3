import tensorflow as tf

def gpugrowth(gpu_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    peinr(len(gpus), "GPUs")
    
    
    
    