#ifndef GLOBAL_MATTING_H
#define GLOBAL_MATTING_H

#include "opencv2/core.hpp"

void expansionOfKnownRegions(cv::InputArray img, cv::InputOutputArray trimap, int niter = 9);
void globalMatting(cv::InputArray image, cv::InputArray trimap, cv::OutputArray foreground, cv::OutputArray alpha, cv::OutputArray conf = cv::noArray());

namespace gs
{
  
   CV_EXPORTS_W void alphamatte(cv::InputArray image,cv::InputArray trimap,cv::OutputArray foreground,cv::OutputArray alpha);
}

#endif
