# Global-Sampling-Method-for-Alpha-Matting
This is a C++ implementation of "A Global Sampling Method for Alpha Matting" by Kaiming He, Christoph Rhemann, Carsten Rother, Xiaoou Tang,JianSun presented at CVPR 2011.


### Steps ###
Use the following command to run the program

 ```g++ pkg-config --cflags opencv globalmatting.cpp pkg-config --libs opencv -o matting```

This produces an executable file called matting

To run the executable file, use the following command

  ```./matting <Path to input image> <Path to trimap> <niter>(number of iterations for expansion of known regions[optional])```

### Results ###

After evaluating this implementation on alphamatting.com, the results are almost as good as the original implementation.

Following were the results:

|     Error type              |      Original implementation    | This implementation  |
|     -----------             |      ------------------------   | -------------------  |  
| Sum of absolute differences |       31                        | 31.3                 |
| Mean square error           |       28.3                      | 29.5                 |
| Gradient error              |       25                        | 26.3                 |
| Connectivity error          |       28                        | 36.3                 |


Some of the outputs with of this implementation are as follows :

|    Image                    | Trimap                          | Alpha matte(this implementation)  |
|  --------------             | --------------                  | ------------------------          |
|![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Input/doll.png "img1" ) |![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Trimap/doll.png "trimap1" ) |![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Results/doll.png "results1" ) |
|![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Input/troll.png "img2" ) |![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Trimap/troll.png "trimap2" ) |![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Results/troll.png "results2" ) |
|![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Input/donkey.png "img1" ) |![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Trimap/donkey.png "trimap1" ) |![alt text](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/Global-Sampling-Method-for-Alpha-Matting/Results/donkey.png "results1" ) |
