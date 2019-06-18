# GSOC-Opencv
Progress of alpha matting project

## Alpha Matting

### Description: ### 
    Project aims at generating a large training dataset for alpha matting, training the model and converting the model to ONNX to  
    be used with OpenCV's DNN module and to integrate some of the best computer vision based best alpha matting algorithms into 
    OpenCV.

### Expected Outcomes:
   
  1.    A large dataset which can be used to train models of alpha matting.
   
  2.    Implement the paper: "AlphaGAN Matting" by Sebastian Lultz.
        
  3.    Trained model by the use of the generated dataset
        
  4.    Convert model to ONNX and provide a running example in OpenCV's DNN module
  
  5.    Implement the paper: "A Global Sampling Method for Alpha Matting" by Kaiming He et al.
        Implement the paper: "Designing Effective Inter-Pixel Information Flow for Natural Image Matting" by Yagiz et. al."
        
  6.    Experiments comparing results to existing alpha matting algorithms at alphamatting.com    
    
 ### Resources 
     alphamatting.com Comparison of many methods, datasets etc
    
 ### Skills Required:
      Excellent C++ and Python coding skills, Experience in training deep learning models in Tensorflow/Pytorch, Experience in 
      Computer Vision, Experience in Deep Learning
    
      Mentors: Steven Puttemans,Gholamreza Amayeh,Sunita Nayak
      Difficulty: Medium-Hard
      
 
 ### Updates: 
 
 1. Read and understood the architecture of each paper. Wrote an article explaining the papers
    [Understanding AlphaGAN matting](https://medium.com/vedacv/understanding-alphagan-matting-1dfae112a412?source=friends_link&sk=75a7bbf958afbf92f5dc53a4c5ff10d4),
    [Paper Summary:Global sampling method for Alpha matting](https://medium.com/vedacv/paper-summary-a-global-sampling-method-for-alpha-matting-490a4217eb2)
    
 2.  Learnt how to use gimp, to create the dataset for the deep learning based model.
 
 3.  Implemented Global sampling based method for alpha matting.
 
 4.  Verified the implementation is correct. Checked the implementation on Alphamatting.com. Here are the results.[Link](https://drive.google.com/file/d/12BW2q3kEfAoZtR4P6DRzQYCraMbkUsPX/view?usp=sharing) 
     (Global sampling method-Vedanta is my implementation)
  
 5. Implemented AlphaGAN matting, but the results even when trained on Adobe dataset is not satisfactory.
    #### Working on improving this now
    
 6. Generated alpha matte for about 100 images using gimp.[Link](https://drive.google.com/drive/folders/12esYUgcR5PDl_wfZXF9dBpNwvO_60yL9)   

