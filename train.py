import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model import resnet152,text_model,resnet50
from dataset import *
import random
import numpy as np
from config import *
from torchvision import transforms as T
import torch.nn.functional as F



def get_resnet152(pretrained=False,num_classes=9):
    img_model = resnet152(pretrained)
    img_model.fc = nn.Linear(2048, num_classes)
    return img_model


def get_resnet50(pretrained=False,num_classes=9,lstm_layers=3):
    img_model = resnet50(pretrained,num_classes,lstm_layers)
    #img_model.fc = nn.Linear(2048, num_classes)
    return img_model


class focal_loss(nn.Module):
    def __init__(self, focusing_param=2, balance_param=1):
        super(focal_loss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, targets):
        targets_num = (targets >-1).sum()  # [N,H,W]
        logpt = -F.cross_entropy(output, targets,reduction='none')
        pt = torch.exp(logpt)

        focal_loss =( -((1 - pt) ** self.focusing_param) * logpt).sum()

        balanced_focal_loss = self.balance_param * focal_loss/targets_num

        return balanced_focal_loss



def main():
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = True

    img_model = get_resnet50(True,opt.num_classes)
    #img_model = text_model(opt.num_classes)

    if opt.resume:
        img_model.load_state_dict(torch.load(opt.weight))

    train_image_path, val_image_path, train_label, val_label = baidu_Urban_image_list(opt.train_list,
                                                                                val=opt.val_rate,
                                                                                seed=SEED)
    train_transfrom = T.Compose([T.RandomResizedCrop(opt.img_size,(0.75,1.5)),
                                 T.RandomHorizontalFlip(),
                                 T.RandomRotation(1),
                                 T.ToTensor(),
                                 T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

    train_dataset = baidu_Urban_loader(train_image_path, train_label, "train", train_transfrom)

    val_transfrom = T.Compose([T.Resize(opt.img_size),
                                 T.ToTensor(),
                                 T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

    val_dataset = baidu_Urban_loader(val_image_path, val_label, "val", val_transfrom)
    train_dataloader = DataLoader(train_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers,
                            shuffle=True,  # disable rectangular training if True
                            pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers,
                            shuffle=False,  # disable rectangular training if True
                            pin_memory=True)
    img_model = nn.DataParallel(img_model.cuda())
    weight = torch.Tensor([0.038, 0.049, 0.102, 0.269, 0.105, 0.066, 0.104, 0.140, 0.127])
    #crit = focal_loss(2,weight)
    crit = nn.CrossEntropyLoss(weight=weight).cuda()
    optimper = torch.optim.SGD(img_model.parameters(),opt.lr,momentum=opt.momen,weight_decay=opt.decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimper,opt.ep_step,0.5)
    best_acc = []
    for ep in range(opt.epochs):
        img_model.train()
        scheduler.step()
        for i,(image,visit,label) in enumerate(train_dataloader):
            optimper.zero_grad()
            image = image.cuda()
            visit = visit.cuda()
            label = label.cuda()
            pred,hidden = img_model(image,visit)
            #pred, hidden = img_model(visit)
            loss = crit(pred,label)
            loss.backward()
            optimper.step()
            if (i+1)%10==0:
                print("epochs: {},iter:{},loss: {}".format(ep+1,i+1,float(loss)))
        if (ep+1) % opt.val_epochs ==0:
            img_model.eval()
            correct = 0
            for i,(image,visit,label) in enumerate(val_dataloader):
                image = image.cuda()
                label = label.cuda()
                n = label.shape[0]
                pred,hidden = img_model(image,visit)
                #pred, hidden = img_model(visit)
                _,predict = pred.topk(1,1)
                predict = predict.t()
                correct += float(torch.sum(predict.eq(label)))/n

            acc = float(correct)/(i+1)
            print("epochs: {},acc: {}".format(ep+1,float(acc)))
            best_acc.append(acc)
            if acc==max(best_acc):
                torch.save(img_model.module.state_dict(),"{}/ep{}_acc{}.pth".format(opt.ckpt,ep+1,acc))
            img_model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs')
    parser.add_argument('--val_epochs', type=int, default=VAL_EPOCHS, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='inference size (pixels)')
    parser.add_argument('--seed', type=int, default=SEED, help='random seed')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    parser.add_argument('--num_classes', default=CLASSES, type=int)
    parser.add_argument('--num_workers', default=NUM_WORKERS, type=int)
    parser.add_argument('--train_list', default=TRAIN_LIST, type=str)
    parser.add_argument('--val_rate', default=VAL_RATE, type=float)
    parser.add_argument('--lr', default=LR, type=float)
    parser.add_argument('--momen', default=MOMEN, type=float)
    parser.add_argument('--decay', default=DECAY, type=float)
    parser.add_argument('--ep_step', default=EP_STEP, type=int)
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--ckpt',default='checkpoint',type=str )
    parser.add_argument('--weight',default='checkpoint.pth',type=str )
    opt = parser.parse_args()

    if not os.path.exists(opt.ckpt):
        os.makedirs(opt.ckpt)
    main()
