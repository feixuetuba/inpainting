import torch
from models.NN.DeepFill2.modules import SNConvWithActivation
from utils import get_pad

nn = torch.nn

class SnPatchGanDirciminator(nn.Module):
    def __init__(self, in_channels=6, basic_num=64):
        super(SnPatchGanDirciminator, self).__init__()
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(in_channels, basic_num, 5, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(basic_num, 2*basic_num, 5, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(2*basic_num, 4*basic_num, 5, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(4*basic_num, 4*basic_num, 5, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(4*basic_num, 4*basic_num, 5, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(4*basic_num, 4*basic_num, 5, 2, padding=get_pad(8, 5, 2)),
        )
        # self.linear = nn.Linear(4*basic_num*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        # x = self.linear(x)
        return x