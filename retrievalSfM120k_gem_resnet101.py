import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.nn import functional as F
from collections import OrderedDict
from torch.autograd import Variable
from scipy.io import loadmat

import torchvision.transforms as transforms

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class RetrievalSfM120k_gem_resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta = { 'std': [1, 1, 1],
                     'imageSize': [224, 224, 3, 1]}

        self.mean_image = 114.039
        self.conv1_relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.res2a_branch2a_relu = nn.ReLU()
        self.res2a_branch2b_relu = nn.ReLU()
        self.res2a_relu = nn.ReLU()
        self.res2b_branch2a_relu = nn.ReLU()
        self.res2b_branch2b_relu = nn.ReLU()
        self.res2b_relu = nn.ReLU()
        self.res2c_branch2a_relu = nn.ReLU()
        self.res2c_branch2b_relu = nn.ReLU()
        self.res2c_relu = nn.ReLU()
        self.res3a_branch2a_relu = nn.ReLU()
        self.res3a_branch2b_relu = nn.ReLU()
        self.res3a_relu = nn.ReLU()
        self.res3b1_branch2a_relu = nn.ReLU()
        self.res3b1_branch2b_relu = nn.ReLU()
        self.res3b1_relu = nn.ReLU()
        self.res3b2_branch2a_relu = nn.ReLU()
        self.res3b2_branch2b_relu = nn.ReLU()
        self.res3b2_relu = nn.ReLU()
        self.res3b3_branch2a_relu = nn.ReLU()
        self.res3b3_branch2b_relu = nn.ReLU()
        self.res3b3_relu = nn.ReLU()
        self.res4a_branch2a_relu = nn.ReLU()
        self.res4a_branch2b_relu = nn.ReLU()
        self.res4a_relu = nn.ReLU()
        self.res4b1_branch2a_relu = nn.ReLU()
        self.res4b1_branch2b_relu = nn.ReLU()
        self.res4b1_relu = nn.ReLU()
        self.res4b2_branch2a_relu = nn.ReLU()
        self.res4b2_branch2b_relu = nn.ReLU()
        self.res4b2_relu = nn.ReLU()
        self.res4b3_branch2a_relu = nn.ReLU()
        self.res4b3_branch2b_relu = nn.ReLU()
        self.res4b3_relu = nn.ReLU()
        self.res4b4_branch2a_relu = nn.ReLU()
        self.res4b4_branch2b_relu = nn.ReLU()
        self.res4b4_relu = nn.ReLU()
        self.res4b5_branch2a_relu = nn.ReLU()
        self.res4b5_branch2b_relu = nn.ReLU()
        self.res4b5_relu = nn.ReLU()
        self.res4b6_branch2a_relu = nn.ReLU()
        self.res4b6_branch2b_relu = nn.ReLU()
        self.res4b6_relu = nn.ReLU()
        self.res4b7_branch2a_relu = nn.ReLU()
        self.res4b7_branch2b_relu = nn.ReLU()
        self.res4b7_relu = nn.ReLU()
        self.res4b8_branch2a_relu = nn.ReLU()
        self.res4b8_branch2b_relu = nn.ReLU()
        self.res4b8_relu = nn.ReLU()
        self.res4b9_branch2a_relu = nn.ReLU()
        self.res4b9_branch2b_relu = nn.ReLU()
        self.res4b9_relu = nn.ReLU()
        self.res4b10_branch2a_relu = nn.ReLU()
        self.res4b10_branch2b_relu = nn.ReLU()
        self.res4b10_relu = nn.ReLU()
        self.res4b11_branch2a_relu = nn.ReLU()
        self.res4b11_branch2b_relu = nn.ReLU()
        self.res4b11_relu = nn.ReLU()
        self.res4b12_branch2a_relu = nn.ReLU()
        self.res4b12_branch2b_relu = nn.ReLU()
        self.res4b12_relu = nn.ReLU()
        self.res4b13_branch2a_relu = nn.ReLU()
        self.res4b13_branch2b_relu = nn.ReLU()
        self.res4b13_relu = nn.ReLU()
        self.res4b14_branch2a_relu = nn.ReLU()
        self.res4b14_branch2b_relu = nn.ReLU()
        self.res4b14_relu = nn.ReLU()
        self.res4b15_branch2a_relu = nn.ReLU()
        self.res4b15_branch2b_relu = nn.ReLU()
        self.res4b15_relu = nn.ReLU()
        self.res4b16_branch2a_relu = nn.ReLU()
        self.res4b16_branch2b_relu = nn.ReLU()
        self.res4b16_relu = nn.ReLU()
        self.res4b17_branch2a_relu = nn.ReLU()
        self.res4b17_branch2b_relu = nn.ReLU()
        self.res4b17_relu = nn.ReLU()
        self.res4b18_branch2a_relu = nn.ReLU()
        self.res4b18_branch2b_relu = nn.ReLU()
        self.res4b18_relu = nn.ReLU()
        self.res4b19_branch2a_relu = nn.ReLU()
        self.res4b19_branch2b_relu = nn.ReLU()
        self.res4b19_relu = nn.ReLU()
        self.res4b20_branch2a_relu = nn.ReLU()
        self.res4b20_branch2b_relu = nn.ReLU()
        self.res4b20_relu = nn.ReLU()
        self.res4b21_branch2a_relu = nn.ReLU()
        self.res4b21_branch2b_relu = nn.ReLU()
        self.res4b21_relu = nn.ReLU()
        self.res4b22_branch2a_relu = nn.ReLU()
        self.res4b22_branch2b_relu = nn.ReLU()
        self.res4b22_relu = nn.ReLU()
        self.res5a_branch2a_relu = nn.ReLU()
        self.res5a_branch2b_relu = nn.ReLU()
        self.res5a_relu = nn.ReLU()
        self.res5b_branch2a_relu = nn.ReLU()
        self.res5b_branch2b_relu = nn.ReLU()
        self.res5b_relu = nn.ReLU()
        self.res5c_branch2a_relu = nn.ReLU()
        self.res5c_branch2b_relu = nn.ReLU()
        self.res5c_relu = nn.ReLU()
        self.pooldescriptor = GeM()
        self.l2descriptor = L2Norm()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.res2a_branch1 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res2a_branch2a = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.res2a_branch2b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res2a_branch2c = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res2b_branch2a = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.res2b_branch2b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res2b_branch2c = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res2c_branch2a = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.res2c_branch2b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res2c_branch2c = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res3a_branch1 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        self.res3a_branch2a = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2))
        self.res3a_branch2b = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res3a_branch2c = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.res3b1_branch2a = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.res3b1_branch2b = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res3b1_branch2c = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.res3b2_branch2a = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.res3b2_branch2b = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res3b2_branch2c = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.res3b3_branch2a = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.res3b3_branch2b = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res3b3_branch2c = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.res4a_branch1 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        self.res4a_branch2a = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2))
        self.res4a_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4a_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b1_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b1_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b1_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b2_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b2_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b2_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b3_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b3_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b3_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b4_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b4_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b4_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b5_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b5_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b5_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b6_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b6_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b6_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b7_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b7_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b7_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b8_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b8_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b8_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b9_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b9_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b9_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b10_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b10_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b10_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b11_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b11_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b11_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b12_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b12_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b12_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b13_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b13_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b13_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b14_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b14_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b14_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b15_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b15_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b15_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b16_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b16_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b16_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b17_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b17_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b17_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b18_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b18_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b18_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b19_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b19_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b19_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b20_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b20_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b20_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b21_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b21_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b21_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res4b22_branch2a = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.res4b22_branch2b = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res4b22_branch2c = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.res5a_branch1 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        self.res5a_branch2a = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2))
        self.res5a_branch2b = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res5a_branch2c = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
        self.res5b_branch2a = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
        self.res5b_branch2b = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res5b_branch2c = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
        self.res5c_branch2a = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
        self.res5c_branch2b = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.res5c_branch2c = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, input):

        conv1 = self.conv1(input)
        check = conv1.data.cpu().numpy()[0].transpose(1, 2, 0)
        conv1xxx = self.conv1_relu(conv1)
        pool1 = self.pool1(conv1xxx)
        res2a_branch1 = self.res2a_branch1(pool1)
        res2a_branch2a = self.res2a_branch2a(pool1)

        res2a_branch2axxx = self.res2a_branch2a_relu(res2a_branch2a)
        res2a_branch2b = self.res2a_branch2b(res2a_branch2axxx)
        res2a_branch2bxxx = self.res2a_branch2b_relu(res2a_branch2b)
        res2a_branch2c = self.res2a_branch2c(res2a_branch2bxxx)
        res2a = torch.add(res2a_branch1, 1, res2a_branch2c)
        res2ax = self.res2a_relu(res2a)

        res2b_branch2a = self.res2b_branch2a(res2ax)

        res2b_branch2axxx = self.res2b_branch2a_relu(res2b_branch2a)
        res2b_branch2b = self.res2b_branch2b(res2b_branch2axxx)

        res2b_branch2bxxx = self.res2b_branch2b_relu(res2b_branch2b)
        res2b_branch2c = self.res2b_branch2c(res2b_branch2bxxx)


        res2b = torch.add(res2ax, 1, res2b_branch2c)
        res2bx = self.res2b_relu(res2b)

        res2c_branch2a = self.res2c_branch2a(res2bx)
        res2c_branch2axxx = self.res2c_branch2a_relu(res2c_branch2a)
        res2c_branch2b = self.res2c_branch2b(res2c_branch2axxx)

        res2c_branch2bxxx = self.res2c_branch2b_relu(res2c_branch2b)

        res2c_branch2c = self.res2c_branch2c(res2c_branch2bxxx)

        res2c = torch.add(res2bx, 1, res2c_branch2c)
        res2cx = self.res2c_relu(res2c)

        res3a_branch1 = self.res3a_branch1(res2cx)
        res3a_branch2a = self.res3a_branch2a(res2cx)

        res3a_branch2axxx = self.res3a_branch2a_relu(res3a_branch2a)

        res3a_branch2b = self.res3a_branch2b(res3a_branch2axxx)
        res3a_branch2bxxx = self.res3a_branch2b_relu(res3a_branch2b)

        res3a_branch2c = self.res3a_branch2c(res3a_branch2bxxx)
        res3a = torch.add(res3a_branch1, 1, res3a_branch2c)
        res3ax = self.res3a_relu(res3a)

        res3b1_branch2a = self.res3b1_branch2a(res3ax)
        res3b1_branch2axxx = self.res3b1_branch2a_relu(res3b1_branch2a)

        res3b1_branch2b = self.res3b1_branch2b(res3b1_branch2axxx)
        res3b1_branch2bxxx = self.res3b1_branch2b_relu(res3b1_branch2b)

        res3b1_branch2c = self.res3b1_branch2c(res3b1_branch2bxxx)
        res3b1 = torch.add(res3ax, 1, res3b1_branch2c)

        res3b1x = self.res3b1_relu(res3b1)

        res3b2_branch2a = self.res3b2_branch2a(res3b1x)
        res3b2_branch2axxx = self.res3b2_branch2a_relu(res3b2_branch2a)

        res3b2_branch2b = self.res3b2_branch2b(res3b2_branch2axxx)
        res3b2_branch2bxxx = self.res3b2_branch2b_relu(res3b2_branch2b)
        res3b2_branch2c = self.res3b2_branch2c(res3b2_branch2bxxx)

        res3b2 = torch.add(res3b1x, 1, res3b2_branch2c)
        res3b2x = self.res3b2_relu(res3b2)

        res3b3_branch2a = self.res3b3_branch2a(res3b2x)
        res3b3_branch2axxx = self.res3b3_branch2a_relu(res3b3_branch2a)

        res3b3_branch2b = self.res3b3_branch2b(res3b3_branch2axxx)
        res3b3_branch2bxxx = self.res3b3_branch2b_relu(res3b3_branch2b)

        res3b3_branch2c = self.res3b3_branch2c(res3b3_branch2bxxx)

        res3b3 = torch.add(res3b2x, 1, res3b3_branch2c)
        res3b3x = self.res3b3_relu(res3b3)


        res4a_branch1 = self.res4a_branch1(res3b3x)
        res4a_branch2a = self.res4a_branch2a(res3b3x)

        res4a_branch2axxx = self.res4a_branch2a_relu(res4a_branch2a)
        res4a_branch2b = self.res4a_branch2b(res4a_branch2axxx)
        res4a_branch2bxxx = self.res4a_branch2b_relu(res4a_branch2b)
        res4a_branch2c = self.res4a_branch2c(res4a_branch2bxxx)

        res4a = torch.add(res4a_branch1, 1, res4a_branch2c)
        res4ax = self.res4a_relu(res4a)

        res4b1_branch2a = self.res4b1_branch2a(res4ax)
        res4b1_branch2axxx = self.res4b1_branch2a_relu(res4b1_branch2a)

        res4b1_branch2b = self.res4b1_branch2b(res4b1_branch2axxx)

        res4b1_branch2bxxx = self.res4b1_branch2b_relu(res4b1_branch2b)
        res4b1_branch2c = self.res4b1_branch2c(res4b1_branch2bxxx)
        res4b1 = torch.add(res4ax, 1, res4b1_branch2c)
        res4b1x = self.res4b1_relu(res4b1)

        res4b2_branch2a = self.res4b2_branch2a(res4b1x)
        res4b2_branch2axxx = self.res4b2_branch2a_relu(res4b2_branch2a)
        res4b2_branch2b = self.res4b2_branch2b(res4b2_branch2axxx)
        res4b2_branch2bxxx = self.res4b2_branch2b_relu(res4b2_branch2b)
        res4b2_branch2c = self.res4b2_branch2c(res4b2_branch2bxxx)
        res4b2 = torch.add(res4b1x, 1, res4b2_branch2c)

        res4b2x = self.res4b2_relu(res4b2)
        res4b3_branch2a = self.res4b3_branch2a(res4b2x)
        res4b3_branch2axxx = self.res4b3_branch2a_relu(res4b3_branch2a)

        res4b3_branch2b = self.res4b3_branch2b(res4b3_branch2axxx)
        res4b3_branch2bxxx = self.res4b3_branch2b_relu(res4b3_branch2b)

        res4b3_branch2c = self.res4b3_branch2c(res4b3_branch2bxxx)

        res4b3 = torch.add(res4b2x, 1, res4b3_branch2c)
        res4b3x = self.res4b3_relu(res4b3)
        res4b4_branch2a = self.res4b4_branch2a(res4b3x)
        res4b4_branch2axxx = self.res4b4_branch2a_relu(res4b4_branch2a)

        res4b4_branch2b = self.res4b4_branch2b(res4b4_branch2axxx)
        res4b4_branch2bxxx = self.res4b4_branch2b_relu(res4b4_branch2b)
        res4b4_branch2c = self.res4b4_branch2c(res4b4_branch2bxxx)
        res4b4 = torch.add(res4b3x, 1, res4b4_branch2c)
        res4b4x = self.res4b4_relu(res4b4)

        res4b5_branch2a = self.res4b5_branch2a(res4b4x)
        res4b5_branch2axxx = self.res4b5_branch2a_relu(res4b5_branch2a)

        res4b5_branch2b = self.res4b5_branch2b(res4b5_branch2axxx)
        res4b5_branch2bxxx = self.res4b5_branch2b_relu(res4b5_branch2b)

        res4b5_branch2c = self.res4b5_branch2c(res4b5_branch2bxxx)
        res4b5 = torch.add(res4b4x, 1, res4b5_branch2c)
        res4b5x = self.res4b5_relu(res4b5)

        res4b6_branch2a = self.res4b6_branch2a(res4b5x)
        res4b6_branch2axxx = self.res4b6_branch2a_relu(res4b6_branch2a)

        res4b6_branch2b = self.res4b6_branch2b(res4b6_branch2axxx)
        res4b6_branch2bxxx = self.res4b6_branch2b_relu(res4b6_branch2b)

        res4b6_branch2c = self.res4b6_branch2c(res4b6_branch2bxxx)
        res4b6 = torch.add(res4b5x, 1, res4b6_branch2c)
        res4b6x = self.res4b6_relu(res4b6)

        res4b7_branch2a = self.res4b7_branch2a(res4b6x)
        res4b7_branch2axxx = self.res4b7_branch2a_relu(res4b7_branch2a)

        res4b7_branch2b = self.res4b7_branch2b(res4b7_branch2axxx)
        res4b7_branch2bxxx = self.res4b7_branch2b_relu(res4b7_branch2b)

        res4b7_branch2c = self.res4b7_branch2c(res4b7_branch2bxxx)
        res4b7 = torch.add(res4b6x, 1, res4b7_branch2c)
        res4b7x = self.res4b7_relu(res4b7)

        res4b8_branch2a = self.res4b8_branch2a(res4b7x)
        res4b8_branch2axxx = self.res4b8_branch2a_relu(res4b8_branch2a)

        res4b8_branch2b = self.res4b8_branch2b(res4b8_branch2axxx)
        res4b8_branch2bxxx = self.res4b8_branch2b_relu(res4b8_branch2b)

        res4b8_branch2c = self.res4b8_branch2c(res4b8_branch2bxxx)
        res4b8 = torch.add(res4b7x, 1, res4b8_branch2c)
        res4b8x = self.res4b8_relu(res4b8)

        res4b9_branch2a = self.res4b9_branch2a(res4b8x)
        res4b9_branch2axxx = self.res4b9_branch2a_relu(res4b9_branch2a)

        res4b9_branch2b = self.res4b9_branch2b(res4b9_branch2axxx)
        res4b9_branch2bxxx = self.res4b9_branch2b_relu(res4b9_branch2b)

        res4b9_branch2c = self.res4b9_branch2c(res4b9_branch2bxxx)
        res4b9 = torch.add(res4b8x, 1, res4b9_branch2c)
        res4b9x = self.res4b9_relu(res4b9)


        res4b10_branch2a = self.res4b10_branch2a(res4b9x)

        res4b10_branch2axxx = self.res4b10_branch2a_relu(res4b10_branch2a)
        res4b10_branch2b = self.res4b10_branch2b(res4b10_branch2axxx)

        res4b10_branch2bxxx = self.res4b10_branch2b_relu(res4b10_branch2b)
        res4b10_branch2c = self.res4b10_branch2c(res4b10_branch2bxxx)
        res4b10 = torch.add(res4b9x, 1, res4b10_branch2c)
        res4b10x = self.res4b10_relu(res4b10)

        res4b11_branch2a = self.res4b11_branch2a(res4b10x)

        res4b11_branch2axxx = self.res4b11_branch2a_relu(res4b11_branch2a)
        res4b11_branch2b = self.res4b11_branch2b(res4b11_branch2axxx)

        res4b11_branch2bxxx = self.res4b11_branch2b_relu(res4b11_branch2b)
        res4b11_branch2c = self.res4b11_branch2c(res4b11_branch2bxxx)

        res4b11 = torch.add(res4b10x, 1, res4b11_branch2c)
        res4b11x = self.res4b11_relu(res4b11)

        res4b12_branch2a = self.res4b12_branch2a(res4b11x)

        res4b11 = torch.add(res4b10x, 1, res4b11_branch2c)
        res4b11x = self.res4b11_relu(res4b11)
        res4b12_branch2axxx = self.res4b12_branch2a_relu(res4b12_branch2a)

        res4b12_branch2b = self.res4b12_branch2b(res4b12_branch2axxx)
        res4b12_branch2bxxx = self.res4b12_branch2b_relu(res4b12_branch2b)

        res4b12_branch2c = self.res4b12_branch2c(res4b12_branch2bxxx)
        res4b12 = torch.add(res4b11x, 1, res4b12_branch2c)
        res4b12x = self.res4b12_relu(res4b12)

        res4b13_branch2a = self.res4b13_branch2a(res4b12x)

        res4b13_branch2axxx = self.res4b13_branch2a_relu(res4b13_branch2a)

        res4b13_branch2b = self.res4b13_branch2b(res4b13_branch2axxx)
        res4b13_branch2bxxx = self.res4b13_branch2b_relu(res4b13_branch2b)

        res4b13_branch2c = self.res4b13_branch2c(res4b13_branch2bxxx)
        res4b13 = torch.add(res4b12x, 1, res4b13_branch2c)
        res4b13x = self.res4b13_relu(res4b13)

        res4b14_branch2a = self.res4b14_branch2a(res4b13x)
        res4b14_branch2axxx = self.res4b14_branch2a_relu(res4b14_branch2a)

        res4b14_branch2b = self.res4b14_branch2b(res4b14_branch2axxx)
        res4b14_branch2bxxx = self.res4b14_branch2b_relu(res4b14_branch2b)
        res4b14_branch2c = self.res4b14_branch2c(res4b14_branch2bxxx)

        res4b14 = torch.add(res4b13x, 1, res4b14_branch2c)
        res4b14x = self.res4b14_relu(res4b14)

        res4b15_branch2a = self.res4b15_branch2a(res4b14x)
        res4b15_branch2axxx = self.res4b15_branch2a_relu(res4b15_branch2a)

        res4b15_branch2b = self.res4b15_branch2b(res4b15_branch2axxx)
        res4b15_branch2bxxx = self.res4b15_branch2b_relu(res4b15_branch2b)

        res4b15_branch2c = self.res4b15_branch2c(res4b15_branch2bxxx)
        res4b15 = torch.add(res4b14x, 1, res4b15_branch2c)
        res4b15x = self.res4b15_relu(res4b15)

        res4b16_branch2a = self.res4b16_branch2a(res4b15x)

        res4b16_branch2axxx = self.res4b16_branch2a_relu(res4b16_branch2a)

        res4b16_branch2b = self.res4b16_branch2b(res4b16_branch2axxx)
        res4b16_branch2bxxx = self.res4b16_branch2b_relu(res4b16_branch2b)
        res4b16_branch2c = self.res4b16_branch2c(res4b16_branch2bxxx)
        res4b16 = torch.add(res4b15x, 1, res4b16_branch2c)
        res4b16x = self.res4b16_relu(res4b16)
        res4b17_branch2a = self.res4b17_branch2a(res4b16x)

        res4b17_branch2axxx = self.res4b17_branch2a_relu(res4b17_branch2a)
        res4b17_branch2b = self.res4b17_branch2b(res4b17_branch2axxx)
        res4b17_branch2bxxx = self.res4b17_branch2b_relu(res4b17_branch2b)
        res4b17_branch2c = self.res4b17_branch2c(res4b17_branch2bxxx)
        res4b17 = torch.add(res4b16x, 1, res4b17_branch2c)
        res4b17x = self.res4b17_relu(res4b17)

        res4b18_branch2a = self.res4b18_branch2a(res4b17x)
        res4b18_branch2axxx = self.res4b18_branch2a_relu(res4b18_branch2a)
        res4b18_branch2b = self.res4b18_branch2b(res4b18_branch2axxx)
        res4b18_branch2bxxx = self.res4b18_branch2b_relu(res4b18_branch2b)

        res4b18_branch2c = self.res4b18_branch2c(res4b18_branch2bxxx)
        res4b18 = torch.add(res4b17x, 1, res4b18_branch2c)
        res4b18x = self.res4b18_relu(res4b18)

        res4b19_branch2a = self.res4b19_branch2a(res4b18x)

        res4b19_branch2axxx = self.res4b19_branch2a_relu(res4b19_branch2a)
        res4b19_branch2b = self.res4b19_branch2b(res4b19_branch2axxx)
        res4b19_branch2bxxx = self.res4b19_branch2b_relu(res4b19_branch2b)
        res4b19_branch2c = self.res4b19_branch2c(res4b19_branch2bxxx)
        res4b19 = torch.add(res4b18x, 1, res4b19_branch2c)
        res4b19x = self.res4b19_relu(res4b19)

        res4b20_branch2a = self.res4b20_branch2a(res4b19x)
        res4b20_branch2axxx = self.res4b20_branch2a_relu(res4b20_branch2a)
        res4b20_branch2b = self.res4b20_branch2b(res4b20_branch2axxx)
        res4b20_branch2bxxx = self.res4b20_branch2b_relu(res4b20_branch2b)
        res4b20_branch2c = self.res4b20_branch2c(res4b20_branch2bxxx)
        res4b20 = torch.add(res4b19x, 1, res4b20_branch2c)
        res4b20x = self.res4b20_relu(res4b20)

        res4b21_branch2a = self.res4b21_branch2a(res4b20x)
        res4b21_branch2axxx = self.res4b21_branch2a_relu(res4b21_branch2a)
        res4b21_branch2b = self.res4b21_branch2b(res4b21_branch2axxx)
        res4b21_branch2bxxx = self.res4b21_branch2b_relu(res4b21_branch2b)
        res4b21_branch2c = self.res4b21_branch2c(res4b21_branch2bxxx)
        res4b21 = torch.add(res4b20x, 1, res4b21_branch2c)
        res4b21x = self.res4b21_relu(res4b21)
        res4b22_branch2a = self.res4b22_branch2a(res4b21x)

        res4b22_branch2axxx = self.res4b22_branch2a_relu(res4b22_branch2a)
        res4b22_branch2b = self.res4b22_branch2b(res4b22_branch2axxx)
        res4b22_branch2bxxx = self.res4b22_branch2b_relu(res4b22_branch2b)
        res4b22_branch2c = self.res4b22_branch2c(res4b22_branch2bxxx)
        res4b22 = torch.add(res4b21x, 1, res4b22_branch2c)
        res4b22x = self.res4b22_relu(res4b22)

        res5a_branch1 = self.res5a_branch1(res4b22x)
        res5a_branch2a = self.res5a_branch2a(res4b22x)
        res5a_branch2axxx = self.res5a_branch2a_relu(res5a_branch2a)
        res5a_branch2b = self.res5a_branch2b(res5a_branch2axxx)
        res5a_branch2bxxx = self.res5a_branch2b_relu(res5a_branch2b)
        res5a_branch2c = self.res5a_branch2c(res5a_branch2bxxx)
        res5a = torch.add(res5a_branch1, 1, res5a_branch2c)
        res5ax = self.res5a_relu(res5a)

        res5b_branch2a = self.res5b_branch2a(res5ax)
        res5b_branch2axxx = self.res5b_branch2a_relu(res5b_branch2a)
        res5b_branch2b = self.res5b_branch2b(res5b_branch2axxx)
        res5b_branch2bxxx = self.res5b_branch2b_relu(res5b_branch2b)
        res5b_branch2c = self.res5b_branch2c(res5b_branch2bxxx)
        res5b = torch.add(res5ax, 1, res5b_branch2c)
        res5bx = self.res5b_relu(res5b)

        res5c_branch2a = self.res5c_branch2a(res5bx)
        res5c_branch2axxx = self.res5c_branch2a_relu(res5c_branch2a)
        res5c_branch2b = self.res5c_branch2b(res5c_branch2axxx)
        res5c_branch2bxxx = self.res5c_branch2b_relu(res5c_branch2b)
        res5c_branch2c = self.res5c_branch2c(res5c_branch2bxxx)

        res5c = torch.add(res5bx, 1, res5c_branch2c)
        xx0 = self.res5c_relu(res5c)

        pooldescriptor = self.pooldescriptor(xx0)
        pooldescriptor = pooldescriptor.view(pooldescriptor.size(0), -1)
        l2descriptor = self.l2descriptor(pooldescriptor)

        return l2descriptor

def retrievalSfM120k_gem_resnet101(weights_path=None, **kwargs):
    """
    load imported model instance
    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = RetrievalSfM120k_gem_resnet101()
    if weights_path:
        state_dict = torch.load(weights_path)
        new_values = rename_dict_keys(state_dict)
        model.load_state_dict(new_values)
    return model

def rename_dict_keys(state_dict):

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace('[','').replace(']','').replace('\'','')
        if 'pooldescriptor' in key:
            key = 'pooldescriptor.p'
        new_state_dict[key] = value
    return new_state_dict


class Normalize(object):

    def __init__(self, mean_image):
        self.mean_image = mean_image

    def __call__(self, img):
        img = np.array(img) - self.mean_image
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        return img


def check_pytorch_model_produces_same_result_as_matconvnet():

    weights_path = 'weights/retrievalSfM120k_gem_resnet101.pth'
    model = retrievalSfM120k_gem_resnet101(weights_path).cuda()

    test_transformer = transforms.Compose([
        Normalize(model.mean_image)
    ])

    image = Image.open('test/oxford_img_1.jpg')
    image = test_transformer(image).unsqueeze(0)
    img_var = Variable(image).cuda()
    res = model(img_var)
    final = res.data.cpu().numpy()[0]
    l2_descriptor_from_matconvnet = loadmat('test/l2descriptor.mat')['scores'].squeeze()
    assert (l2_descriptor_from_matconvnet.round(4) == final.round(4)).all()

if __name__ == '__main__':
    check_pytorch_model_produces_same_result_as_matconvnet()