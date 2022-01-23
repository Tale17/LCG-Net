import torch.nn as nn
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from function import patch_mv_norm, patch_adain

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(513, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

winsize = 7

class PSF(nn.Module):
    def __init__(self, in_planes):
        super(PSF, self).__init__()
        self.e = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self,front, back, mask):
        EE = self.e(patch_adain(front, back, winsize))
        #Structural feature
        FF = self.f(patch_mv_norm(front,winsize))
        GG = self.g(patch_mv_norm(back,winsize))
        #Appearance feature
        HH = self.h(back)
        b, _, h, w = GG.size()
        FF = FF.view(b, -1, w * h)
        GG = GG.view(b, -1, w * h)
        F_n = (FF*FF).sum(dim=1).sqrt()
        G_n = (GG * GG).sum(dim=1).sqrt()
        S = torch.mul(FF, GG).sum(dim=1) / (F_n*G_n)
        S_n = ((S - S.min(dim=1)[0].unsqueeze(1)) / (S.max(dim=1)[0].unsqueeze(1) - S.min(dim=1)[0].unsqueeze(1))).view(
            b, 1, h, w)
        O = torch.cat((torch.mul(S_n, EE) + torch.mul(1 - S_n, HH), mask), dim=1)
        return O

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.PSF = PSF(in_planes = 512)

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def encode_with_intermediate_a(self, input):
        results = [input]
        for i in range(4,12):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def tv_loss(self,target):
        assert (len(target.size()) == 4)
        dx = (target[:, :, 1:, :-1] - target[:, :, :-1, :-1]) ** 2
        dy = (target[:, :, :-1, 1:] - target[:, :, :-1, :-1]) ** 2
        dxy = dx + dy
        tv = dxy.sum()
        return tv

    def normalize(self, a):
        b = np.zeros(a.shape)
        for k in range(a.shape[0]):
            max = np.max(a[k])
            min = np.min(a[k])
            m = a[k].shape[0]
            n = a[k].shape[1]
            for i in range(m):
                for j in range(n):
                    b[k, i, j] = (a[k, i, j] - min) / (max - min + 1e-5)
        return b

    def attention(self,f):
        n, c, w, h = f.shape
        ff = np.zeros((n,1, w, h))
        af = np.zeros((n,1, w, h))
        for i in range(c):
            ff[:,0,:,:] += f[:, i, :, :]
        ff = ff / c  # n,w,h
        for i in range(f.shape[0]):
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            a, b = saliency.computeSaliency(ff[i,0])
            af[i,0] = (b - b.min()) / (b.max() - b.min())
        return af

    def remove_loss(self, input, target,a_n):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        a_n = a_n.view(1,-1)
        d = input.size(1)
        input = input.transpose(0, 1).contiguous().view(d, -1)
        target = target.transpose(0, 1).contiguous().view(d, -1)
        input = torch.mul(input, a_n)
        target = torch.mul(target, a_n)
        return self.mse_loss(input, target)

    def bp_loss(self, x,y, n):
        (b, c, h, w) = x.size()
        zeroPad2d = torch.nn.ZeroPad2d(n // 2)

        x_pad = zeroPad2d(x)
        x_fold = F.unfold(x_pad, (n, n), stride=1).view(b, c, n * n, -1)
        x_mean = x_fold.mean(dim=2)
        x_std = x_fold.var(dim=2).sqrt()

        y_pad = zeroPad2d(y)
        y_fold = F.unfold(y_pad, (n, n), stride=1).view(b, c, n * n, -1)
        y_mean = y_fold.mean(dim=2)
        y_std = y_fold.var(dim=2).sqrt()
        return self.mse_loss(x_mean, y_mean) + \
               self.mse_loss(x_std, y_std)

    def imm_loss(self,input,target,a,mask):
        assert (input.size() == target.size())
        n,c,w,h = input.shape
        maskk = (mask.view(n,1,1,w*h) + mask.view(n,1,w*h,1)).clip(0,1)#只关注那些都在mask里的点
        input = input.view(n,c,1,w*h) - input.view(n,c,w*h,1)
        target = target.view(n,c,1,w*h) - target.view(n,c,w*h,1)
        aa = a.view(n,1,1,w*h) + a.view(n,1,w*h,1)
        return self.mse_loss(input * aa * maskk, target * aa * maskk)

    def forward(self, front, back, mask):
        device = torch.device("cuda:0")
        back_feats = self.encode_with_intermediate(back)
        front_feat = self.encode(front)
        front_feats = self.encode_with_intermediate(front)
        mask_down = F.interpolate(mask, scale_factor=1/8, mode="nearest")

        a = torch.from_numpy(self.attention(front_feat.cpu().numpy())).float().to(device)
        t = self.PSF(front_feat, back_feats[-1], mask_down)
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_cam = self.imm_loss(g_t_feats[-1], front_feat,a,mask_down)

        nnn = winsize
        loss_bp = self.bp_loss(g_t_feats[0],back_feats[0],nnn)
        loss_tv = self.tv_loss(g_t)
        a = torch.from_numpy(self.attention(front_feats[0].cpu().numpy())).float().to(device)
        a_n = 1-a
        remove = self.remove_loss(g_t_feats[0], back_feats[0], a_n)
        for i in range(1, 4):
            loss_bp += self.bp_loss(g_t_feats[i], back_feats[i],nnn)
            a = torch.from_numpy(self.attention(front_feats[i].cpu().numpy())).float().to(device)
            a_n = 1 - a
            remove += self.remove_loss(g_t_feats[i], back_feats[i], a_n)
        return loss_cam, loss_bp, remove, loss_tv