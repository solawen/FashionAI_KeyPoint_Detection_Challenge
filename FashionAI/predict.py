
import pandas as pd
import os
import cv2
from scipy.io import loadmat, savemat
import skimage.io as io
import random
import numpy as np
from src.tool import get_lanmark_from_heatmap, get_lanmark_from_heatmap_by_filter
import sys
import time
from multiprocessing import Process, JoinableQueue, Queue
import math


def get_time_str():
    # ex:'2018_03_30_20_34'
    return time.strftime("%Y_%m_%d_%H_%M", time.localtime((time.time())))


t0 = time.time()
f_path = 'test2/'
#VAL_NUM = 4000
skirt_points = ['image_id', 'waistband_left',
                'waistband_right', 'hemline_left', 'hemline_right']
trousers_points = ['image_id', 'waistband_left', 'waistband_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
blouse_points = ['image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'center_front', 'armpit_left',
                 'armpit_right', 'top_hem_left', 'top_hem_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out']
outwear_points = ['image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                  'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']
dress_points = ['image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'center_front', 'armpit_left', 'armpit_right', 'waistline_left',
                'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right']
skirt_col = [15, 16, 17, 18]
trousers_col = [15, 16, 19, 20, 21, 22, 23]
blouse_col = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
outwear_col = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
dress_col = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
col = {'blouse': blouse_col, 'outwear': outwear_col,
       'skirt': skirt_col, 'trousers': trousers_col, 'dress': dress_col}


from keras.models import load_model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tensorflow as tf
from keras import backend as k
###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

from src.CPN_model import custom_loss


def get_img_and_landmark(df):
    image_category = np.array(df['image_category'])
    img_name_array = np.array(df['image_id'])
    return img_name_array, image_category


def predictor(landmark_queue, t0, image_category, img_name_array, batch_size=20):
    # generate heatmap
    data_len = image_category.shape[0]
    output_array = np.array(
        ['-1_-1_-1' for _i in range(26)], dtype='<U70').reshape(1, -1)
    print("data lenth is :%d" % data_len)
    counter = 0
    for i in range(math.ceil(data_len/batch_size)):
        result = landmark_queue.get(timeout=1800)
        result_num = result['landmark'].shape[0]
        for j in range(result_num):
            output_array[counter, 0] = img_name_array[result['id'][j]]
            output_array[counter, 1] = image_category[result['id'][j]]
            for k in range(24):
                if result['landmark'][j, k, 0] != 0:
                    output_array[counter, k+2] = str(result['landmark'][j, k, 0]) + '_' + str(
                        result['landmark'][j, k, 1]) + '_1'
            counter += 1
            output_array = np.append(output_array, np.array(
                ['-1_-1_-1' for _i in range(26)]).reshape(1, -1), axis=0)
        out_str = "Processing %dth batches,total:%d \r" % (i, int(data_len/20))
        sys.stdout.write(out_str)
        sys.stdout.flush()
        #print('%d output_array processed'%i)
        landmark_queue.task_done()
    output_array = output_array[:-1, :]
    df_result = pd.DataFrame(output_array, columns=pd.read_csv(
        'train/Annotations/train.csv').columns)
    df_result.to_csv('result_one_model_%s.csv' % get_time_str(), index=False)
    print(" ")
    print("time:%f" % (time.time()-t0))


def get_landmarker(predict_queue, landmark_queue, point_num=24, heatmap_size=128):
    # image_category 20*1
    skirt_col = [15, 16, 17, 18]
    trousers_col = [15, 16, 19, 20, 21, 22, 23]
    blouse_col = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
    outwear_col = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    dress_col = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
    col = {'blouse': blouse_col, 'outwear': outwear_col,
           'skirt': skirt_col, 'trousers': trousers_col, 'dress': dress_col}
    Threhold = 0.5
    tmp_padding = np.zeros((1024, 1024))
    K = np.array([[np.linalg.norm([64 - _x, 64 - _y])
                   for _x in range(129)] for _y in range(129)])
    K[K <= 64] = 1
    K[K > 64] = 0
    flip_point_order = [1,0,2,4,3,6,5,8,7,11,12,9,10,14,13,16,15,18,17,19,22,23,20,21]
    while True:
        pass
        data_pre = predict_queue.get(timeout=1800)
        start_time = time.time()
        #use original and flipped heatmap to get more accurate key point
        Y_pre = data_pre['heatmap']
        image_category = data_pre['image_category']
        batch_size = Y_pre.shape[0]
        landmark = np.zeros((batch_size, point_num, 2), dtype=int)
        Y_pre_flip = data_pre['heatmap_flip'][:,:,:,flip_point_order]

        for i in range(batch_size):
            now_Y_pre = Y_pre[i, :, :,
                              col[image_category[i]]].transpose((1, 2, 0))
            now_Y_pre_flip = Y_pre_flip[i, :, :,
                              col[image_category[i]]].transpose((1, 2, 0))
            tmp_landmark = np.zeros((now_Y_pre.shape[2], 2), dtype=int)
            for j in range(now_Y_pre.shape[2]):
                p1 = np.array(now_Y_pre[:, :, j]).reshape(
                    heatmap_size, heatmap_size)
                p1_flip = cv2.flip(np.array(now_Y_pre_flip[:, :, j]).reshape(
                    heatmap_size, heatmap_size), 1)
                p1_combine = (p1+p1_flip)/2
                tmp_map = cv2.resize(p1_combine, (512, 512))
                tmp_padding[256:256+512, 256:256+512] = tmp_map
                blur = cv2.filter2D(tmp_padding, -1, K)
                tmp_landmark[j, 0], tmp_landmark[j, 1] = (
                    np.argmax(blur) % 1024 - 256, np.argmax(blur)//1024 - 256)
            landmark[i, col[image_category[i]], :] = tmp_landmark
        landmark_queue.put({'id': data_pre['id'], 'landmark': landmark})
        predict_queue.task_done()


def img_reader(image_queue, img_name_array, data_len, batch_size):
    for i in range(math.ceil(data_len/batch_size)):
        # read img
        img_array = np.ones((batch_size, 512, 512, 3))
        img_flip_array = np.ones((batch_size, 512, 512, 3))
        for j in range(batch_size):
            if (i*batch_size+j) >= data_len:
                img_array = img_array[:j, :, :, :]
                img_flip_array = img_flip_array[:j, :, :, :]
                break
            img_o = cv2.imread(
                "test2/"+img_name_array[i*batch_size+j].strip(), cv2.IMREAD_COLOR)/127.5 - 1
            img_o = img_o[:, :, ::-1]
            H, W, C = img_o.shape
            img_array[j, 0:H, 0:W, :] = img_o
            img_flip_array[j, :, :, :] = cv2.flip(img_array[j, :, :, :], 1)
        image_queue.put(
            {"img_array": img_array, "img_flip_array": img_flip_array}, timeout=1800)


def main(model_to_load):
    pass
    df = pd.read_csv(f_path+'test.csv')
    # shuffle to get faster processing time
    df = df.sample(frac=1).reset_index(drop=True)
    img_name_array, image_category = get_img_and_landmark(df)
    data_len = image_category.shape[0]
    batch_size_o = 20
    image_queue = JoinableQueue(3)
    predict_queue = JoinableQueue(16)
    landmark_queue = JoinableQueue(8)
    model = load_model(model_to_load,
                       {'custom_loss': custom_loss})
    print("model read completed")

    p_i = Process(target=img_reader, args=(image_queue, img_name_array,
                                           data_len, batch_size_o))
    p_i.daemon = True
    p_i.start()

    c_pool = []
    for _i in range(8):
        c = Process(target=get_landmarker,
                    args=(predict_queue, landmark_queue))
        c.daemon = True
        c.start()
        c_pool.append(c)

    p1 = Process(target=predictor,
                 args=(landmark_queue, t0, image_category, img_name_array))
    p1.daemon = True

    p1.start()

    for i in range(math.ceil(data_len/batch_size_o)):
        img_array = image_queue.get(timeout=1800)
        batch_size = img_array["img_array"].shape[0]
        tmp_out_1 = model.predict(
            img_array["img_array"], batch_size=batch_size)
        tmp_out_flip = model.predict(
            img_array["img_flip_array"], batch_size=batch_size)
        image_queue.task_done()
        predict_queue.put({'id': [i for i in range(i*batch_size_o, i*batch_size_o+batch_size)], 'heatmap': tmp_out_1[0],
                           'heatmap_flip': tmp_out_flip[0], 'image_category': image_category[i*batch_size_o:i*batch_size_o+batch_size]}, timeout=1800)

    p1.join()


if __name__ == '__main__':
    model_to_load = 'model/CPN_ONE_v02018_04_23_18_30.model'
    main(model_to_load)
