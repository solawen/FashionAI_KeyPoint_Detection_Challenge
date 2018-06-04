import pandas as pd
from scipy.io import loadmat,savemat
import numpy as np
df_path1 = 'train/Annotations/train.csv'
df_path2 = 'train/Annotations/annotations.csv'
df_path3 = 'test/fashionAI_key_points_test_a_answer_20180426.csv'
out_path = 'train/Annotations/allv2.mat'



if __name__ == '__main__':
    df1 = pd.read_csv(df_path1)
    df2 = pd.read_csv(df_path2)
    df3 = pd.read_csv(df_path3)
    df_all = (pd.concat([df1,df2,df3],ignore_index = True).reset_index(drop = True)).sample(frac=1).reset_index(drop=True)

    tmp_label=np.array(df_all)
    landmark_label = []
    visibility_label = []
    image_id=[]
    for row_index,row in enumerate(tmp_label):
        tmp_row_landmark = []
        tmp_row_visibility = []
        for col in row[2:]:
            tmp_visibility = [0,0,0]
            tmp_label = col.strip().split('_')
            tmp_visibility[int(tmp_label[2])+1]=1
            tmp_row_landmark.extend([np.float32(tmp_label[0])/512,np.float32(tmp_label[1])/512])
            tmp_row_visibility.extend(tmp_visibility)
        landmark_label.append(tmp_row_landmark)
        visibility_label.append(tmp_row_visibility)
        image_id.append(row[0])
    landmark_label = np.array(landmark_label)
    visibility_label = np.array(visibility_label)
    savemat(out_path,{'image_id':image_id,'landmark_label':landmark_label,'visibility_label':visibility_label})