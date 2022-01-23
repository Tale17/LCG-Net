import torch
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    if(len(size) == 4):
        B, C = size[:2]
        feat_var = feat.view(B, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        return feat_mean, feat_std
    if len(size)==5:
        B, C, N = size[:3]
        feat_var = feat.view(B, C, N, -1).var(dim=3) + eps
        feat_std = feat_var.sqrt().view(B, C, N, 1, 1)
        feat_mean = feat.view(B, C, N, -1).mean(dim=3).view(B, C, N, 1, 1)
        return feat_mean, feat_std

def mean_var_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    norm_feat = (feat - mean.expand(size)) / std.expand(size)
    return norm_feat

def patch_mv_norm(feat,n):
    (b,c,h,w) = feat.size()
    eps = 1e-5
    zeroPad2d = torch.nn.ZeroPad2d(n//2)
    feat_pad = zeroPad2d(feat)
    feat_fold = F.unfold(feat_pad, (n, n), stride=1).view(b,c,n*n,-1)
    feat_mean = feat_fold.mean(dim=2)
    feat_std = feat_fold.var(dim=2).sqrt() + eps
    feat_norm = ((feat.view(b,c,-1) - feat_mean) / feat_std).view(b,c,h,w)
    return feat_norm

def patch_adain(x,y,n):
    (b, c, h, w) = y.size()
    zeroPad2d = torch.nn.ZeroPad2d(n // 2)
    x_norm = patch_mv_norm(x,n)

    y_pad = zeroPad2d(y)
    y_fold = F.unfold(y_pad, (n, n), stride=1).view(b, c, n * n, -1)
    y_mean = y_fold.mean(dim=2).view(b,c,h,w)
    y_std = y_fold.var(dim=2).sqrt().view(b,c,h,w)
    x_adain = x_norm * y_std + y_mean
    return x_adain