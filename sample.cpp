#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <cv.h>
 
using namespace std;
using namespace cv;
using namespace std;
 
int main() {
 
	cvNamedWindow("Brezel detecting camera", 1);
	// Capture images from any camera connected to the system
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
 
	// Load the trained model
	CascadeClassifier pedsDetector;
	pedsDetector.load("src/brezel.xml");
 
	if (pedsDetector.empty()) {
		printf("Empty model.");
		return 0;
	}
 
	char key;
	while (true) {
 
		// Get a frame from the camera
		Mat frame = cvQueryFrame(capture);
 
		std::vector<Rect> peds;
 
		// Detect peds
		pedsDetector.detectMultiScale(frame, peds, 1.1, 30,
				0 | CV_HAAR_SCALE_IMAGE, Size(200, 320));
 
		for (int i = 0; i < (int) peds.size(); i++) {
			Point pt1(peds[i].x, peds[i].y);
			Point pt2(peds[i].x + peds[i].width,
					peds[i].y + peds[i].width);
 
			// Draw a rectangle around the detected brezel
			rectangle(frame, pt1, pt2, Scalar(0, 0, 255), 2);
			putText(frame, "Brezel", pt1, FONT_HERSHEY_PLAIN, 1.0,
					Scalar(255, 0, 0), 2.0);
 
		}
 
		// Show the transformed frame
		imshow("Brezel detecting camera", frame);
 
		// Read keystrokes, exit after ESC pressed
		key = cvWaitKey(10);
		if (char(key) == 27) {
			break;
		}
	}
 
	return 0;
}
