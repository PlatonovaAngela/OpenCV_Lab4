#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    Mat past1 = imread("1.jpg");
    namedWindow("Past1");
    imshow("Past1",past1);
    waitKey();
    destroyWindow("Past1");

    Mat past2 = imread("2.jpg");
    namedWindow("Past2");
    imshow("Past2",past2);
    waitKey();
    destroyWindow("Past2");

    int minHessian = 400;
    Ptr<cv::xfeatures2d::SURF> detect =  xfeatures2d::SURF::create(minHessian);
    vector<KeyPoint> key1, key2;
    Mat descriptor1, descriptor2;
    detect->detectAndCompute(past1,Mat(),key1,descriptor1);
    detect->detectAndCompute(past2,Mat(),key2,descriptor2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptor1, descriptor2, knn_matches, 2 );
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
           if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
           {
               good_matches.push_back(knn_matches[i][0]);
           }
    }

    std::vector<Point2f> past1r;
    std::vector<Point2f> past2r;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        past1r.push_back( key1[ good_matches[i].queryIdx ].pt );
        past2r.push_back( key2[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography(past1r,past2r, RANSAC );

    int width = past1.cols;
    int height = past1.rows;
    Mat result;
    warpPerspective(past1,result,H,Size(width,height));
    for(int i = 0; ;++i)
    {
        if(result.at<Vec3b>(0,i)[0] == 0
           && result.at<Vec3b>(0,i)[1] == 0
           && result.at<Vec3b>(0,i)[2] == 0)
        {
            width = i;
            break;
        }
    }
    int width_2 = past2.cols - width;
    Mat res;
    Mat res2(past2,Rect(width-1,0,width_2, past2.rows));
    res.create(height,past1.cols + width_2,past1.type());
    past1.copyTo(res(Rect(0,0,past1.cols,past1.rows)));
    res2.copyTo(res(Rect(past1.cols-1,0,res2.cols,res2.rows)));
    namedWindow("Panorama");
    imshow("Panorama",res);
    waitKey();
    destroyAllWindows();

    return 0;
}
