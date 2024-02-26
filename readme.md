# CamShow

## Introduction

This is a platform for rat behavior data collection and analysis.

## Directory

```
camshow
├── dist  // Save the generated .exe file
├── RatNet
│   │── do_conv_pytorch.py
│   │── Models.py
│   │── RatNetAttention_DOConv.py
│   │── models.pth  // Pre-trained weights.
├── Saved-test  // Default folder for saving results and videos
├── CamShow.py  // The main function. Each major module is briefly commented.
├── ObardCamDisp.py  // UI interface file.
├── RatNet1.py  // Pose estimation network model and post-processing, called at line 575 in Camshow.py.
```