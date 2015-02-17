#include <iostream>
#include <typeinfo>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

void LBPConvert(Mat inputFrame,Mat outputFrame) {
    int x,y;
    for (x=0;x<inputFrame.size().width;x++) {
        outputFrame.at<unsigned char>(0,x)=0;
        outputFrame.at<unsigned char>(inputFrame.size().height-1,x)=0;
    }
    for (y=1;y<inputFrame.size().height-1;y++) {
        outputFrame.at<unsigned char>(y,0)=0;
        outputFrame.at<unsigned char>(y,inputFrame.size().width-1)=0;
        for (x=1;x<inputFrame.size().width-1;x++) {
            unsigned char pixCount = 0;
            unsigned char pixValue = inputFrame.at<unsigned char>(y,x);
            pixCount |= ((inputFrame.at<unsigned char>(y-1,x-1) - pixValue)&0x80)>>7;
            pixCount |= ((inputFrame.at<unsigned char>(y,x-1) - pixValue)&0x80)>>6;
            pixCount |= ((inputFrame.at<unsigned char>(y+1,x-1) - pixValue)&0x80)>>5;
            pixCount |= ((inputFrame.at<unsigned char>(y+1,x) - pixValue)&0x80)>>4;
            pixCount |= ((inputFrame.at<unsigned char>(y+1,x+1) - pixValue)&0x80)>>3;
            pixCount |= ((inputFrame.at<unsigned char>(y,x+1) - pixValue)&0x80)>>2;
            pixCount |= ((inputFrame.at<unsigned char>(y-1,x+1) - pixValue)&0x80)>>1;
            pixCount |= ((inputFrame.at<unsigned char>(y-1,x) - pixValue)&0x80);
            outputFrame.at<unsigned char>(y,x) = pixCount;
        }
    }
}


int main()
{
    VideoCapture cap(0);

    namedWindow("Texture",CV_WINDOW_AUTOSIZE);



    bool running = true;
    int state = 1;
    while (running) {
        int key = waitKey(10);
        switch (key) {
            case '1':
                state = 1;
                break;
            case '2':
                state = 2;
                break;
            case '3':
                state = 3;
                break;
            case 27:
            case 'q':
                running=false;
                break;
            default:
                break;
        }

        Mat frame;
        cap.read(frame);
        Mat bwframe(frame.size(),CV_8U);
        Mat lbpframe(frame.size(),CV_8U);

        switch (state) {
            case 1:
                imshow("Texture",frame);
                break;
            case 2:
                cvtColor(frame,bwframe,CV_BGR2GRAY);
                imshow("Texture",bwframe);
                break;
            case 3:
                cvtColor(frame,bwframe,CV_BGR2GRAY);
                LBPConvert(bwframe,lbpframe);
                imshow("Texture",lbpframe);
                break;
            default:
                state=1;
                break;
        }
    }
    return 0;
}
