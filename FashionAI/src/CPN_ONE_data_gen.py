import time
import random
import os
from scipy.io import loadmat
import random
import numpy as np
import cv2

def get_heatmap(label_size, map_value, x, y, Radius):
    _x = int(round(x * label_size))
    _y = int(round(y * label_size))
    _x = min(_x,label_size-1)
    _y = min(_y,label_size-1)
    heatmap = np.array(
                    map_value[label_size-_y:label_size*2-_y, label_size-_x:label_size*2-_x])
    heatmap[heatmap <= Radius] = 1
    heatmap[heatmap > Radius] = 0
    return heatmap

def random_img(img,landmark):
    #input: img 512*512*3 input image
    #landmark point_num*2 landmark
    #return: img_crop: 512*512*3 image after rotation and crop
    #landmark_crop: corresponding landmark of img_crop
    if random.random()>0.5:
        flip_point_order = [1,0,2,4,3,6,5,8,7,11,12,9,10,14,13,16,15,18,17,19,22,23,20,21]
        img = cv2.flip(img, 1)
        tmp_l = np.array(landmark)
        tmp_l[:,0] = 1 - tmp_l[:,0]
        landmark = tmp_l[flip_point_order,:]
    tmp_row_landmark = landmark[:,:]*512+256
    tmp_row_landmark[tmp_row_landmark<256] = 512
    tmp_img = np.ones((1024,1024,3))
    H,W,C = img.shape
    tmp_img[256:256+H,256:256+W,:] = img
    rows,cols = tmp_img.shape[:2]
    rota_angel = ((360+random.randint(-15,15))%360)
    M = cv2.getRotationMatrix2D((512,512), rota_angel, 1.0)
    img_prepro = cv2.warpAffine(tmp_img,M,(rows,cols))
    landmark_prepro = np.asarray([(M[0][0]*x+M[0][1]*y+M[0][2],
                    M[1][0]*x+M[1][1]*y+M[1][2]) for (x, y) in tmp_row_landmark])
    min_x = (np.max(landmark_prepro[:,0])) - 512
    min_y = (np.max(landmark_prepro[:,1])) - 512
    max_x = (np.min(landmark_prepro[:,0]))
    max_y = (np.min(landmark_prepro[:,1]))
    gap_x = (max_x-min_x)/3
    gap_y = (max_y-min_y)/3
    if (gap_x<0)or(gap_y<0):
        return img,landmark
    min_x += gap_x
    max_x -= gap_x
    min_y += gap_y
    max_y -= gap_y
    crop_x = random.randint(int(min_x),int(max_x))
    crop_y = random.randint(int(min_y),int(max_y))
    img_crop = img_prepro[crop_y:crop_y+512,crop_x:crop_x+512,:]
    landmark_crop = np.zeros_like(landmark_prepro)
    landmark_crop[:,0] = landmark_prepro[:,0] - crop_x
    landmark_crop[:,1] = landmark_prepro[:,1] - crop_y
    return img_crop,landmark_crop/512

def get_data_gen(batch_size, mat_path, img_root, Radius, Val_num,point_num = 24, mode = 'Train', label_size = [128,64,32,16]):
    mat = loadmat(mat_path)
    #get data with random preprocessing
    if mode == 'Train':
        img_id = mat['image_id'][Val_num:]
        landmark = mat['landmark_label'][Val_num:]
        vis = mat['visibility_label'][Val_num:]
        s = np.arange(len(img_id))
        np.random.shuffle(s)
        img_id = img_id[s]
        landmark = landmark[s]
        vis = vis[s]
    else:
        img_id = mat['image_id'][:Val_num]
        landmark = mat['landmark_label'][:Val_num]
        vis = mat['visibility_label'][:Val_num]

    landmark = landmark.reshape((-1, point_num, 2))
    vis = np.argmax(vis.reshape((-1,point_num, 3)), axis=2) - 1
    data_len = img_id.shape[0] - 1
    BATCH_IMG = np.ones((batch_size, 512,512,3))
    BATCH_P2 = np.zeros((batch_size,label_size[0],label_size[0],point_num))
    BATCH_P3 = np.zeros((batch_size,label_size[1],label_size[1],point_num))
    BATCH_P4 = np.zeros((batch_size,label_size[2],label_size[2],point_num))
    BATCH_P5 = np.zeros((batch_size,label_size[3],label_size[3],point_num))
    counter = 0
    map_value_P2 = np.array([[np.linalg.norm([label_size[0] - _x, label_size[0] - _y])
                          for _x in range(label_size[0] * 2)] for _y in range(label_size[0] * 2)])
    map_value_P3 = np.array([[np.linalg.norm([label_size[1] - _x, label_size[1] - _y])
                          for _x in range(label_size[1] * 2)] for _y in range(label_size[1] * 2)])
    map_value_P4 = np.array([[np.linalg.norm([label_size[2] - _x, label_size[2] - _y])
                          for _x in range(label_size[2] * 2)] for _y in range(label_size[2] * 2)])
    map_value_P5 = np.array([[np.linalg.norm([label_size[3] - _x, label_size[3] - _y])
                          for _x in range(label_size[3] * 2)] for _y in range(label_size[3] * 2)])
    while True:
        pass
        for i in range(data_len):
            img_o = cv2.imread(img_root+img_id[i].strip(),
                             cv2.IMREAD_COLOR)/127.5 - 1
            img_o = img_o[:,:,::-1]
            H,W,C = img_o.shape
            now_img = np.ones((512,512,3))
            now_img[0:H,0:W,:] = img_o
            now_landmark = landmark[i,:,:]
            if (mode == 'Train')and(random.random()>0.5):
                now_img, now_landmark = random_img(now_img,landmark[i,:,:])
            BATCH_IMG[counter,:,:,:] = now_img
            for j,(x,y) in enumerate(now_landmark):
                if vis[i,j] == 1:
                    # get heatmap
                    BATCH_P2[counter,:,:,j] = get_heatmap(label_size[0],map_value_P2,x,y,Radius[0])
                    BATCH_P3[counter,:,:,j] = get_heatmap(label_size[1],map_value_P3,x,y,Radius[1])
                    BATCH_P4[counter,:,:,j] = get_heatmap(label_size[2],map_value_P4,x,y,Radius[2])
                    BATCH_P5[counter,:,:,j] = get_heatmap(label_size[3],map_value_P5,x,y,Radius[3])
                else:
                    BATCH_P2[counter,:,:,j] = np.zeros((label_size[0],label_size[0]))
                    BATCH_P3[counter,:,:,j] = np.zeros((label_size[1],label_size[1]))
                    BATCH_P4[counter,:,:,j] = np.zeros((label_size[2],label_size[2]))
                    BATCH_P5[counter,:,:,j] = np.zeros((label_size[3],label_size[3]))
            if counter==batch_size-1:
                yield(BATCH_IMG,  {'final_output':BATCH_P2, 'pred_p2': BATCH_P2, 'pred_p3': BATCH_P3, 'pred_p4': BATCH_P4, 'pred_p5': BATCH_P5})
                counter = 0
            else:
                counter += 1