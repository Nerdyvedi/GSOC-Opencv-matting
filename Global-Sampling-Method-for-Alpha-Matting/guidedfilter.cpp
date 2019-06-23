// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include <precomp.hpp>
#include "guidedfilter.h"

//Used to blur the image using the normalized box filter of size r*r
static cv::Mat custom_blur(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class guidedFilterImpl
{
public:
    virtual ~guidedFilterImpl() {}

    cv::Mat filter(const cv::Mat &p, int depth);

protected:
    int Idepth;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class guidedFilterMono : public guidedFilterImpl
{
public:
    guidedFilterMono(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    int r;
    double eps;
    cv::Mat I, mean_I, var_I;
};

class guidedFilterColor : public guidedFilterImpl
{
public:
    guidedFilterColor(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    std::vector<cv::Mat> Ichannels;
    int r;
    double eps;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat guidedFilterImpl::filter(const cv::Mat &p, int depth)
{
    cv::Mat p2 = convertTo(p, Idepth);

    cv::Mat result;
    if (p.channels() == 1)
    {
        result = filterSingleChannel(p2);
    }
    else
    {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        for (std::size_t i = 0; i < pc.size(); ++i)
            pc[i] = filterSingleChannel(pc[i]);

        cv::merge(pc, result);
    }

    return convertTo(result, depth == -1 ? p.depth() : depth);
}

guidedFilterMono::guidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    mean_I = custom_blur(I, r);
    cv::Mat mean_II = custom_blur(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat guidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = custom_blur(p, r);
    cv::Mat mean_Ip = custom_blur(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

    cv::Mat mean_a = custom_blur(a, r);
    cv::Mat mean_b = custom_blur(b, r);

    return mean_a.mul(I) + mean_b;
}

guidedFilterColor::guidedFilterColor(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    cv::Mat I;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    cv::split(I, Ichannels);

    mean_I_r = custom_blur(Ichannels[0], r);
    mean_I_g = custom_blur(Ichannels[1], r);
    mean_I_b = custom_blur(Ichannels[2], r);

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    cv::Mat var_I_rr = custom_blur(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = custom_blur(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = custom_blur(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = custom_blur(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = custom_blur(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = custom_blur(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;

    // Inverse of Sigma + eps * I
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

cv::Mat guidedFilterColor::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = custom_blur(p, r);

    cv::Mat mean_Ip_r = custom_blur(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = custom_blur(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = custom_blur(Ichannels[2].mul(p), r);

    // covariance of (I, p) in each local patch.
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b); // Eqn. (15) in the paper;

    return (custom_blur(a_r, r).mul(Ichannels[0])
          + custom_blur(a_g, r).mul(Ichannels[1])
          + custom_blur(a_b, r).mul(Ichannels[2])
          + custom_blur(b, r));  // Eqn. (16) in the paper;
}


guidedFilter::guidedFilter(const cv::Mat &I, int r, double eps)
{
    CV_Assert(I.channels() == 1 || I.channels() == 3);

    if (I.channels() == 1)
        impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);
    else
        impl_ = new GuidedFilterColor(I, 2 * r + 1, eps);
}

guidedFilter::~guidedFilter()
{
    delete impl_;
}

cv::Mat guidedFilter::filter(const cv::Mat &p, int depth) const
{
    return impl_->filter(p, depth);
}

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth)
{
    return GuidedFilter(I, r, eps).filter(p, depth);
}
