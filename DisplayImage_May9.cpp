#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include<iterator>
#include<algorithm>

using namespace cv;
using namespace std;



void nms(
	const vector<Rect>& srcRects,
	vector<Rect>& resRects,
	float thresh
	)
{
	resRects.clear();

	const size_t size = srcRects.size();
	if (!size)
	{
		return;
	}

	// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
	std::multimap<int, size_t> idxs;
	for (size_t i = 0; i < size; ++i)
	{
		idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
	}

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		const Rect& rect1 = srcRects[lastElem->second];

		resRects.push_back(rect1);

		idxs.erase(lastElem);

		if(idxs.size() > 0){
		for (auto pos = std::begin(idxs); pos != std::end(idxs); )
		{
			// grab the current rectangle
			const Rect& rect2 = srcRects[pos->second];

			float intArea = (rect1 & rect2).area();
			float unionArea = rect1.area() + rect2.area() - intArea;
			float overlap = intArea / unionArea;

			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > thresh)
			{
				pos = idxs.erase(pos);
			}
			else
			{
				++pos;
			}
		}
		}
	}
}






int main(int argc, char** argv )
{

    int total_peds_ref = 0;
    int total_peds_rcv = 0;
    char str1[200];
    char str2[200];
    char str3[200];
    char str4[200];
  
    memset(str1, '\0', sizeof(str1));
    memset(str2, '\0', sizeof(str2));
    memset(str3, '\0', sizeof(str3));
    memset(str4, '\0', sizeof(str4));

    //Object detection code

    CascadeClassifier pedDetector;
    pedDetector.load("cascade15.xml");


    //String path("/home/sbaidya/stream/daimler/TestData/*.pgm");
    //String path("/home/sbaidya/test/TestData/*.pgm");
    //String path("/home/sbaidya/test/rcvd/*.pgm");
    String path("/home/sbaidya/Downloads/campus_frames/*.pgm");
    //String rpath("/home/sbaidya/test/rcv1/*.pgm");
    //String rpath("/home/sbaidya/cliserv_program/images/*.pgm");
    String rpath("/home/sbaidya/wrkshp/images/*.pgm");
    
    vector<String> fn;
    glob(path, fn, true);  
    vector<String> fnr;
    glob(rpath, fnr, true);  

    //while( cap.isOpened()){
    for(size_t k=0; k<std::min(fn.size(), fnr.size()); ++k)
    {

	Mat image = imread(fn[k]);
	Mat imgr = imread(fnr[k]);

	std::vector<Rect> peds;
	std::vector<Rect> pedsr;

	// Detect peds
	//pedDetector.detectMultiScale(image, peds, 1.1, 30, 0 | CV_HAAR_SCALE_IMAGE, Size(200, 320));
	//pedDetector.detectMultiScale(image, peds, 2.5, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(32, 68), Size(192, 408));
	pedDetector.detectMultiScale(image, peds, 2.5, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 50), Size(200, 500));
	pedDetector.detectMultiScale(imgr, pedsr, 2.5, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 50), Size(200, 500));


	vector<Rect> nmsRect;
	nms(peds, nmsRect, 0.7);
	vector<Rect> nmsRectr;
	nms(pedsr, nmsRectr, 0.7);
	
	for (int i = 0; i < (int) nmsRect.size(); i++) 
        {
                Point pt1(nmsRect[i].x, nmsRect[i].y);
                Point pt2(nmsRect[i].x + nmsRect[i].width,nmsRect[i].y + nmsRect[i].height);

                // Draw a rectangle around the detected pedestrian
                rectangle(image, pt1, pt2, Scalar(0, 0, 255), 2);
                putText(image, "Pedestran", pt1, FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2.0);
		total_peds_ref++;

        }


	for (int i = 0; i < (int) nmsRectr.size(); i++) 
        {
                Point pt1r(nmsRectr[i].x, nmsRectr[i].y);
                Point pt2r(nmsRectr[i].x + nmsRectr[i].width,nmsRectr[i].y + nmsRectr[i].height);

                // Draw a rectangle around the detected pedestrian
                rectangle(imgr, pt1r, pt2r, Scalar(0, 0, 255), 2);
                putText(imgr, "Pedestran", pt1r, FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2.0);
		total_peds_rcv++;
        }

	namedWindow("Display Image1", WINDOW_AUTOSIZE );
	moveWindow("Display Image1", 500, 10);
	imshow("Display Image1", image);
	waitKey(25);

	namedWindow("Display Image2", WINDOW_AUTOSIZE );
	moveWindow("Display Image2", 1150, 10);
	imshow("Display Image2", imgr);
	waitKey(25);

    }

        sprintf(str1, "Number of peds. detected ");
	sprintf(str2, "in reference video : %d", total_peds_ref);
        sprintf(str3, "Number of peds. detected ");
	sprintf(str4, "in received video : %d", total_peds_rcv);

	Mat pic1 = Mat::zeros(500, 640, CV_8UC3);
	putText(pic1, str1, Point(80, 200), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255,0), 2.0);
	putText(pic1, str2, Point(80, 250), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255,0), 2.0);
	namedWindow("Image1", WINDOW_AUTOSIZE );
	moveWindow("Image1", 500, 10);
	imshow("Image1", pic1);
	//waitKey(1000000);

	Mat pic2 = Mat::zeros(500, 640, CV_8UC3);
	putText(pic2, str3, Point(80, 200), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255,0), 2.0);
	putText(pic2, str4, Point(80, 250), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255,0), 2.0);
	namedWindow("Image2", WINDOW_AUTOSIZE );
	moveWindow("Image2", 1150, 10);
	imshow("Image2", pic2);
	waitKey(1000000);


    printf("\nTotal number of pedestrian detected in reference video : %d \n", total_peds_ref);
    printf("Total number of pedestrian detected in received video : %d \n", total_peds_rcv);

    return 0;
}




