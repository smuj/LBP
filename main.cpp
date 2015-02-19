#include <iostream>
#include <typeinfo>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

#define BLOCK_SIZE 8
#define BLOCK_SIZE_CONTINUOUS 16

typedef struct label {
    unsigned char value;
    unsigned char position;
} label_t;

void HistConvertContinuous(Mat inputFrame, Mat outputFrame, int blockSize,bool justHue) {
    int x,y;
    for (y=0;y<inputFrame.size().height;y+=1) {
        for (x=0;x<inputFrame.size().width;x+=1) {
            int hist[256]={0};
            int x1,y1;
            for (y1=((y-(blockSize/2))>=0?y-(blockSize/2):0);y1<((y+(blockSize/2))<inputFrame.size().height?y+(blockSize/2):inputFrame.size().height);y1++) {
                for (x1=(x-(blockSize/2))>=0?x-(blockSize/2):0;x1<((x+(blockSize/2))<inputFrame.size().width?x+(blockSize/2):inputFrame.size().width);x1++) {
                    hist[inputFrame.at<unsigned char>(y1,x1)]++;
                }
            }
            int h,hMax=0,hMaxVal=0;
            for (h=1;h<256;h++) {
                if (hist[h]>hMaxVal) {
                    hMax=h;
                    hMaxVal=hist[h];
                }
            }
            if (justHue) {
                outputFrame.at<unsigned char[3]>(y1,x1)[0]=hMax;
                outputFrame.at<unsigned char[3]>(y1,x1)[1]=255;
                outputFrame.at<unsigned char[3]>(y1,x1)[2]=255;
            } else {
                outputFrame.at<unsigned char[3]>(y1,x1)[0]=hMax;
                outputFrame.at<unsigned char[3]>(y1,x1)[1]=255;
                outputFrame.at<unsigned char[3]>(y1,x1)[2]=(unsigned char)(255*(float)hMaxVal/(float)(blockSize*blockSize));
            }
        }
    }
}

void HistConvert(Mat inputFrame, Mat outputFrame, int blockSize,bool justHue) {
    int x,y;
    for (y=0;(y+blockSize)<inputFrame.size().height;y+=blockSize) {
        for (x=0;(x+blockSize)<inputFrame.size().width;x+=blockSize) {
            int hist[256]={0};
            int x1,y1;
            for (y1=y;y1<y+blockSize;y1++) {
                for (x1=x;x1<x+blockSize;x1++) {
                    hist[inputFrame.at<unsigned char>(y1,x1)]++;
                }
            }

            int h,hMax=0,hMaxVal=0;
            for (h=1;h<256;h++) {
                if (hist[h]>hMaxVal) {
                    hMax=h;
                    hMaxVal=hist[h];
                }
            }
            if (justHue) {
                for (y1=y;y1<y+blockSize;y1++) {
                    for (x1=x;x1<x+blockSize;x1++) {
                        outputFrame.at<unsigned char[3]>(y1,x1)[0]=hMax;
                        outputFrame.at<unsigned char[3]>(y1,x1)[1]=255;
                        outputFrame.at<unsigned char[3]>(y1,x1)[2]=255;
                    }
                }
            } else {
                for (y1=y;y1<y+blockSize;y1++) {
                    for (x1=x;x1<x+blockSize;x1++) {
                        outputFrame.at<unsigned char[3]>(y1,x1)[0]=hMax;
                        outputFrame.at<unsigned char[3]>(y1,x1)[1]=255;
                        outputFrame.at<unsigned char[3]>(y1,x1)[2]=(unsigned char)(255*(float)hMaxVal/(float)(blockSize*blockSize));
                    }
                }
            }
        }
    }
}

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

 //   Mat tempImage;
   // tempImage=imread("grass.png", CV_LOAD_IMAGE_GRAYSCALE);

    //namedWindow("Grass", WINDOW_AUTOSIZE);
    //imshow("Grass",tempImage);
    //Mat templbpframe(tempImage.size(),CV_8U);
    //LBPConvert(tempImage,templbpframe);

    //imwrite("grasstexture.png",templbpframe);

    bool running = true;
    bool justHue = true;
    int state = 1;
    while (running) {
        int key = waitKey(30);
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
            case '4':
                state = 4;
                break;
            case '5':
                state = 5;
                break;
            case 'j':
                justHue = true;
                break;
            case 'k':
                justHue = false;
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
        Mat blockframe(frame.size(),CV_8UC3);
        Mat outframe(frame.size(),CV_8UC3);

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
            case 4:
                cvtColor(frame,bwframe,CV_BGR2GRAY);
                LBPConvert(bwframe,lbpframe);
                HistConvert(lbpframe,blockframe,BLOCK_SIZE,justHue);
                cvtColor(blockframe,outframe,CV_HSV2BGR);
                imshow("Texture",outframe);
                break;
            case 5:
                cvtColor(frame,bwframe,CV_BGR2GRAY);
                LBPConvert(bwframe,lbpframe);
                HistConvertContinuous(lbpframe,blockframe,BLOCK_SIZE_CONTINUOUS,justHue);
                cvtColor(blockframe,outframe,CV_HSV2BGR);
                imshow("Texture",outframe);
                break;
            default:
                state=1;
                break;
        }
    }
    return 0;
}
