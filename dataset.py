import torch
import torch.utils.data as data
import os
import glob
import cv2
import numpy as np
import PIL.Image as Image
from sklearn.model_selection import train_test_split,StratifiedKFold


def baidu_Urban_image_list(images_path,val=0.25,seed=42):
    file_folders = os.listdir(images_path)
    #print(file_folders)
    all_path = []
    all_label = []
    for folder in file_folders:
        path = glob.glob(images_path + folder + "/*.jpg")
        for file in path:
            all_path.append(file)
            all_label.append(int((file.split('.')[-2]).split('_')[-1])-1)
    train_image_path,val_image_path,train_label,val_label = train_test_split(all_path,all_label,test_size=val,random_state=seed)
    #sfolder = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)

    return train_image_path,val_image_path,train_label,val_label



class baidu_Urban_loader(data.Dataset):

    def __init__(self, images_paths,image_labels, mode='train',transform_fn=None):

        self.image_data_list = images_paths
        self.image_label_list = image_labels
        self.transform_fn = transform_fn

        print("Total %s examples:"%mode, len(self.image_label_list))

    def __getitem__(self, index):
        images_path = self.image_data_list[index]
        label = self.image_label_list[index]
        image = Image.open(images_path)
        visit = np.load('data/npy/train_visit/'+images_path.split('/')[-1].split('.')[0]+'.npy')
        visit = (visit-visit.mean())/visit.std()
        visit = torch.from_numpy(visit).float()
        #label = torch.from_numpy(label).long()
        if self.transform_fn:
            image = self.transform_fn(image)  # 这里进行预处理

        return image,visit,label

    def __len__(self):
        return len(self.image_data_list)



if __name__=="__main__":
    train_image_path, val_image_path, train_label, val_label = baidu_Urban_image_list('data/baidu_Urban/train/', val=0.25, seed=42)
    print(train_image_path[:30])
    print(train_label[:30])