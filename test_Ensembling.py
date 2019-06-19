import torch
import argparse
import os
import glob
import numpy as np
import cv2
from train import get_resnet152,get_resnet50
import torchvision.transforms.functional as F

def test():
    model1 = get_resnet50(False,9,4)
    model1.load_state_dict(torch.load(opt.weight1))
    model1 = model1.cuda()
    model1.eval()

    model2 = get_resnet50(False,9,3)
    model2.load_state_dict(torch.load(opt.weight2))
    model2 = model2.cuda()
    model2.eval()

    output = open(opt.output,'w')

    all_test_list = glob.glob(opt.test_folder+'*.jpg')

    for image_path in all_test_list:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (opt.img_size,opt.img_size), interpolation=cv2.INTER_CUBIC)
        image = F.to_tensor(image) # hwc-->chw
        image = F.normalize(image,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        image = image.unsqueeze(0).cuda()
        visit = np.load('data/npy/test_visit/'+image_path.split('\\')[-1].split('.')[0]+'.npy')
        visit  = (visit-visit.mean())/visit.std()
        visit = torch.from_numpy(visit).float().unsqueeze(0).cuda()

        pred1,_ = model1(image,visit)
        pred1 = torch.softmax(pred1,1)

        pred2,_ = model2(image,visit)
        pred2 = torch.softmax(pred2,1)

        pred = 0.3*pred1+0.7*pred2

        _, predict = pred.topk(1, 1)
        #print(predict)
        predict = predict.t().squeeze(0)
        for i in predict:
            output.write(image_path.split('.')[-2].split('\\')[-1]+'\t'+'00'+str(int(i+1))+'\n')
    output.close()




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight1',default='checkpoint/ep192_acc0.65875.pth',type=str)
    parser.add_argument('--weight2',default='checkpoint/ep12.pth',type=str)
    parser.add_argument('--test_folder',default='data/baidu_Urban/test/',type=str)
    parser.add_argument('--img_size', default=100, type=int)
    parser.add_argument('--output',default='AreaID.txt',type=str)
    opt = parser.parse_args()
    test()





