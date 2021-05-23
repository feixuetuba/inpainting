import  torch

from models.NN.DeepFill2.modules import GatedConv2dWithActivation, GatedDeConv2dWithActivation, Self_Attn, \
    ContextualAttention
from utils import get_pad

nn = torch.nn

class InpaintSNNet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, in_channels=5, out_channels=3, basic_num=48):
        super(InpaintSNNet, self).__init__()
        self.coarse_net = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(in_channels, basic_num, 5, 1, padding=2),
            # downsample 128
            GatedConv2dWithActivation(basic_num, 2*basic_num, 3, 2, padding=1),
            GatedConv2dWithActivation(2*basic_num, 2*basic_num, 3, 1, padding=1),
            #downsample to 64
            GatedConv2dWithActivation(2*basic_num, 4*basic_num, 3, 2, padding=1),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            # atrous convlution
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=2, padding=2),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=4, padding=4),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=8, padding=8),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=16, padding=16),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            #Self_Attn(4*basic_num, 'relu'),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            # upsample
            GatedDeConv2dWithActivation(2, 4*basic_num, 2*basic_num, 3, 1, padding=1),
            #Self_Attn(2*basic_num, 'relu'),
            GatedConv2dWithActivation(2*basic_num, 2*basic_num, 3, 1, padding=1),
            GatedDeConv2dWithActivation(2, 2*basic_num, basic_num, 3, 1, padding=1),

            GatedConv2dWithActivation(basic_num, basic_num//2, 3, 1, padding=1),
            #Self_Attn(basic_num//2, 'relu'),
            GatedConv2dWithActivation(basic_num//2, out_channels, 3, 1, padding=1, activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(3, basic_num, 5, 1, padding=2),
            # downsample
            GatedConv2dWithActivation(basic_num, basic_num, 3, 2, padding=1),
            GatedConv2dWithActivation(basic_num, 2*basic_num, 3, 1, padding=1),
            # downsample
            GatedConv2dWithActivation(2*basic_num, 2*basic_num, 3, 2, padding=1),
            GatedConv2dWithActivation(2*basic_num, 4*basic_num, 3, 1, padding=1),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=2, padding=2),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=4, padding=4),
            #Self_Attn(4*basic_num, 'relu'),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=8, padding=8),

            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, dilation=16, padding=16)
        )
        # self.refine_attn = Self_Attn(4*basic_num, 'relu', with_attn=False)
        self.refine_attn1 = nn.Sequential(
            GatedConv2dWithActivation(3, basic_num, 5, 1, padding=2),
            GatedConv2dWithActivation(basic_num, basic_num, 3, 2, padding=1),
            GatedConv2dWithActivation(basic_num, 2*basic_num, 3, 1, padding=1),
            GatedConv2dWithActivation(2*basic_num, 4*basic_num, 3, 2, padding=1),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            )

        self.contextAttentation = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                            fuse=True, use_cuda=True, device_ids=[])
        self.refine_attn2 = nn.Sequential(
            GatedConv2dWithActivation(4 * basic_num, 4 * basic_num, 3, 1, padding=1, activation=nn.ReLU()),
            GatedConv2dWithActivation(4 * basic_num, 4 * basic_num, 3, 1, padding=1, activation=nn.ReLU()),
        )
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(8*basic_num, 4*basic_num, 3, 1, padding=1),

            GatedConv2dWithActivation(4*basic_num, 4*basic_num, 3, 1, padding=1),
            GatedDeConv2dWithActivation(2, 4*basic_num, 2*basic_num, 3, 1, padding=1),
            GatedConv2dWithActivation(2*basic_num, 2*basic_num, 3, 1, padding=1),
            GatedDeConv2dWithActivation(2, 2*basic_num, basic_num, 3, 1, padding=1),

            GatedConv2dWithActivation(basic_num, basic_num//2, 3, 1, padding=1),
            #Self_Attn(basic_num, 'relu'),
            GatedConv2dWithActivation(basic_num//2, 3, 3, 1, padding=1, activation=None),
            nn.Tanh()
        )


    def forward(self, masked_imgs, masks, edges=None):
        # Coarse  1 represents masked point
        # masked_imgs =  imgs * (1 - masks) #+ masks
        if edges == None:
            input_imgs = torch.cat([masked_imgs, masks], dim=1)
        else:
            edges = edges * masks
            input_imgs = torch.cat([masked_imgs, edges, torch.full_like(masks, 1.), masks], dim=1)
        #print(input_imgs.size(), imgs.size(), masks.size())
        x = self.coarse_net(input_imgs)
        # x = torch.clamp(x, -1., 1.)
        coarse_x = x
        x_now = x * masks + masked_imgs
        x1 = self.refine_conv_net(x_now)
        x = self.refine_attn1(x_now)
        x, offset_flow = self.contextAttentation(x,x,masks)
        x = self.refine_attn2(x)
        #print(x.size(), attention.size())
        x = torch.cat([x1, x], dim=1)
        x = self.refine_upsample_net(x)
        return coarse_x, x, offset_flow

