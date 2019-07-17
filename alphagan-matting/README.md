# AlphaGAN-Matting
Implementation of AlphaGAN, a Generative adversial method for natural image matting

### Architecture ###

#### Generator ####  
A decoder encoder based architecture is used. 

Encoder is the same as ResNet50 minus the last 2 layers.The implementation of decoder is based on the implementation decoder architecture used in Deep image matting paper.

#### Discriminator ####
The discriminator used here is the PatchGAN discriminator. The implementation here is inspired from the implementation of CycleGAN 
from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


### Results ###

Following are some of the samples generated from this model

|     Image     |     Ground Truth      |     Model Output      |
| ------------- | --------------------  | --------------------- |
|![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/alphagan-matting/AlphaMatting/results/img1.png "img1") | ![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/alphagan-matting/AlphaMatting/results/gt1.png "gt1") |![alt text] (https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/alphagan-matting/AlphaMatting/results/pred1.png "pred1")|
