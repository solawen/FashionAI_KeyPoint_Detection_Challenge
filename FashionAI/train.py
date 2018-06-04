import time
import os
from src.CPN_ONE_model import CPN
import argparse
import tensorflow as tf
from keras import backend as K
from src.CPN_ONE_data_gen import get_data_gen

def get_time_str():
    # ex:'2018_03_30_20_34'
    return time.strftime("%Y_%m_%d_%H_%M", time.localtime((time.time())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--BatchSize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--FREEZE', action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    VAL_NUM = 4000
    time_stamp = get_time_str()

    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################


    nn_train = CPN(weight_decay=1e-4, gpu_num=int(len(args.gpus)/2+1), load_model=args.model)
    train_gen = get_data_gen(batch_size=args.BatchSize, mat_path='train/Annotations/allv2.mat',
                             img_root='train/', point_num=24,
                             Radius=[15, 11, 7, 3], Val_num=VAL_NUM, mode='Train')
    val_gen = get_data_gen(batch_size=args.BatchSize, mat_path='train/Annotations/allv2.mat',
                           img_root='train/', point_num=24,
                           Radius=[15, 11, 7, 3], Val_num=VAL_NUM, mode='Val')
    nn_train.train(train_gen=train_gen, val_gen=val_gen,
                   save_model_path='model/CPN_ONE_v0_1'+time_stamp,
                   data_len=53795, BATCH_SIZE=args.BatchSize, lr_rate=args.lr, FREEZ_BASE_MODEL=args.FREEZE, epochs=10)
