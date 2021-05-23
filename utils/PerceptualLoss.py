import torch
import torchvision
from torch import nn


class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, layers, weights=None):
        super(PerceptualLoss, self).__init__()
        self.blocks = torchvision.models.vgg16(pretrained=True).features.eval()
        # for p in self.block:
        #     p.requires_grad = False
        # self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.weights = weights
        layer_map = {
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
            'conv3_1': 10,
            'conv3_2': 12,
            'conv3_3': 14,
            'conv4_1': 17,
            'conv4_2': 19,
            'conv4_3': 21,
            'conv5_1': 24,
            'conv5_2': 26,
            'conv5_3': 28,
        }
        self.loss_fun = loss_func
        self.layers = [layer_map[_] for _ in layers]
        if weights is not None:
            assert len(weights) == len(self.layers), f"Size of layers:{len(self.layers)} != size of weights:{len(weights)}"
        else:
            self.weights = [1.0] *len(self.layers)
        sorted(self.layers)
        self.blocks = self.blocks[:self.layers[-1]+1]

    def forward(self, x, y):
        y = y.detach()
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)

        y = (y - self.mean) / self.std
        y = self.transform(y, mode='bilinear', size=(224, 224), align_corners=False)

        x_features = []
        y_features = []
        index = 0
        count = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.layers:
                x_features.append(x)
                y_features.append(y)
        for x, y, w in zip(x_features, y_features, self.weights):
            count += (self.loss_fun(x, y) * w)
        return count / len(self.layers)