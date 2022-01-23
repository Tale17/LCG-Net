# Location-Free Camouflage Generation Network

Pytorch implementation of ''Location-Free Camouflage Generation Network''. 

## 📋 Table of content
 1. [📎 Paper Link](#1)
 2. [💡 Abstract](#2)
 3. [✨ Motivation](#3)
 4. [📖 Method](#4)
 6. [📃 Requirements](#5)
 7. [✏️ Usage](#6)
 8. [📊 Experimental Results](#7)
 9. [🍎 Potential Applications](#8)
 10. [✉️ Statement](#9)
 11. [🔍 Citation](#10)

## 📎 Paper Link <a name="1"></a> 
> Location-Free Camouflage Generation Network ([link](https://arxiv.org/pdf/xxxx.xxxxx.pdf))
* Authors: Yangyang Li*, Wei Zhai*, Yang Cao, Zheng-jun Zha
* Institution: University of Science and Technology of China (USTC)

## 💡 Abstract <a name="2"></a> 
xxx

## ✨ Motivation <a name="3"></a> 

<p align="center">
    <img src="./image/motivation.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Motivation.**  Comparison of SOTA method [13] and our results in regions with complex structures and multiple appearances. Part (A): Structural consistency.
Zhang et al. break the continuation of vegetation at the marked position, while we preserve this structure and are therefore more visually natural. Part (B):
Appearance consistency. Zhang et al. mixed multiple appearances of the camouflage region, making the foreground too standout, while we distinguish the
various appearances and achieve the purpose of camouflage.


## 📖 Method <a name="4"></a> 

<p align="center">
    <img src="./image/network.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Overview of Location-free Camouflage Generation Network (LCG-Net).** LCG-Net accepts foreground image If and background image Ib with the
same size. The encoder is taken from the first four layers of VGG-19 for extracting the high-level features. PSF module fuses the foreground and background
features and feeds the result into the decoder to generate an output image Io. We finally embed Io into Ib to get the refined result Ic. We use the same
structure as the Encoder E to calculate the loss functions. The specific structures of the encoder and decoder are given in the supplementary materials.
Position-aligned Structural Fusion (PSF) module. It adaptively fuses the high-level features of the foreground and background according to their point-to-point
structure similarity, and feeds the result to the decoder. Norm represents the normalization operation.

## 📃 Requirements <a name="5"></a> 
  - python 3
  - pytorch 
  - opencv 

## ✏️ Usage <a name="6"></a> 

```
### Download models
Download [models](https://pan.baidu.com/s/1IIaX2CDG-rH2gLjAwhV2TA) (Password: wuj2) and put `decoder.pth`, `PSF.pth`,`vgg_normalised.pth` under `models/`.

### Test


### Train
```

## 📊 Experimental Results <a name="7"></a> 


<p align="center">
    <img src="./image/exp1.png" width="750"/> <br />
    <em> 
    </em>
</p>

<p align="center">
    <img src="./image/exp2.png" width="750"/> <br />
    <em> 
    </em>
</p>

## 🍎 Potential Applications <a name="8"></a>

<p align="center">
    <img src="./image/app1.png" width="750"/> <br />
    <em> 
    </em>
</p>

<p align="center">
    <img src="./image/app2.png" width="450"/> <br />
    <em> 
    </em>
</p>

<p align="center">
    <img src="./image/app3.png" width="450"/> <br />
    <em> 
    </em>
</p>

## ✉️ Statement <a name="9"></a> 
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [lyy1030@mail.ustc.edu.cn](lyy1030@mail.ustc.edu.cn) or [wzhai056@mail.ustc.edu.cn](wzhai056@mail.ustc.edu.cn).


## 🔍 Citation <a name="10"></a> 

```
@inproceedings{Li2022Location,
  title={Location-Free Camouflage Generation Network},
  author={Yangyang Li and Wei Zhai and Yang Cao and Zheng-jun Zha},
  year={2022}
}
```

