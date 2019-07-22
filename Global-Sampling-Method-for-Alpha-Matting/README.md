# Global-Sampling-Method-for-Alpha-Matting
This is a C++ implementation of "A Global Sampling Method for Alpha Matting" by Kaiming He, Christoph Rhemann, Carsten Rother, Xiaoou Tang,JianSun presented at CVPR 2011.

### Results ###

After evaluating this implementation on alphamatting.com, the results are almost as good as the original implementation.

Following were the results:

|     Error type              |      Original implementation    | This implementation  |
|     -----------             |      ------------------------   | -------------------  |  
| Sum of absolute differences |       31                        | 31.3                 |
| Mean square error           |       28.3                      | 29.5                 |
| Gradient error              |       25                        | 26.3                 |
| Connectivity error          |       28                        | 36.3                 |


