import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchgan.layers.selfattention import SelfAttention2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) #[n,2048,1,1]
        #self.fc_my = nn.Linear(512 * block.expansion, num_classes)
        self.lstm = nn.LSTM(24,128,num_layers=3,batch_first=True,dropout=0.2,bidirectional=True)
        self.linear = nn.Linear(46592,2048)

        self.fc_1 = nn.Linear(4096, 1024)
        self.fc_2 = nn.Linear(1024,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))


        return nn.Sequential(*layers)

    def forward(self, x,y,hidden=None):

        if hidden is None:
            h_0 = y.data.new(6,y.size()[0],128).fill_(0).float()
            c_0 = y.data.new(6,y.size()[0],128).fill_(0).float()
        else:
            h_0,c_0 = hidden


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        y = y.permute(0, 2, 1, 3).contiguous() #[n,26,7,24]
        y = y.view(y.size()[0], -1, y.size()[3]) # [n,182,24]
        #y = y.permute(1,0,2).contiguous() #[182.n.24]
        y,hidden = self.lstm(y,( h_0,c_0)) #[182,n,48]
        #y = torch.cat([y[:,0,:],y[:,1,:]],dim=1)
        y = y.contiguous()# [n,182,48]
        y = y.view(y.size()[0],-1) #[n,8736]
        y = self.linear(y) #[n,2048]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = x.view(x.size()[0],-1)
        x = torch.cat((x,y),1)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)

        return x,hidden


def resnet152(pretrained=False, classes=9):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']),strict=False)
    return model

def resnet50(pretrained=False, classes=9):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
    return model


class text_model(nn.Module):
    def __init__(self,num_classes):
        super(text_model,self).__init__()

        self.lstm = nn.LSTM(24,128,num_layers=5,batch_first=True,dropout=0.2,bidirectional=True)
        self.linear = nn.Linear(46592,2048)
        self.relu = nn.ReLU(inplace=True)
        #self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(2048,num_classes)

    def forward(self,y,hidden=None):
        if hidden is None:
            h_0 = y.data.new(10,y.size()[0],128).fill_(0).float()
            c_0 = y.data.new(10,y.size()[0],128).fill_(0).float()
        else:
            h_0,c_0 = hidden
        y = y.permute(0, 2, 1, 3).contiguous()  # [n,26,7,24]
        y = y.view(y.size()[0], -1, y.size()[3])  # [n,182,24]
        # y = y.permute(1,0,2).contiguous() #[182.n.24]
        y, hidden = self.lstm(y, (h_0, c_0))  # [182,n,48]
        #y = torch.cat([y[:, 0, :], y[:, 1, :]], dim=1)
        #y = y.sum(dim=1)
        y = y.contiguous()  # [n,182,48]
        y = y.view(y.size()[0],-1) #[n,8736]
        y = self.linear(y)  # [n,2048]
        #y = self.fc_1(y)
        y = self.relu(y)
        y = self.fc_2(y)

        return y,hidden
