/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <vector>

#define ESC_KEY 27
#define HEIGHT 264
#define WIDTH 581 

char* groundTruthLocation = "Media/GroundTruth.png";
char* compositeLocation = "Media/TestingData.jpg";
char* trainingDataLocation = "Media/Reds.png";

std::string composite(compositeLocation);
std::string groundTruth(groundTruthLocation);
std::string trainingData(trainingDataLocation);

Mat compositeImage = imread(composite, 1);
Mat groundTruthImage = imread(groundTruth, 1);
Mat trainingImage = imread(trainingData, 1);

void generateHistogram();

int main(int argc, const char** argv) {
	generateHistogram();
	while (cv::waitKey(30) != ESC_KEY) {}
	return 0;
}

void generateHistogram() {
	Mat training_hsv, h, hist, backprojection;
	float range[] = { 0, 180 };
	const float* ranges[] = { range };
	int ch[] = { 0, 0 };
	int numBins = 6;
	int histSize = MAX(numBins, 2);

	//get dat training hue sample
	cvtColor(trainingImage, training_hsv, CV_BGR2HSV);
	h.create(training_hsv.size(), training_hsv.depth());
	mixChannels(&training_hsv, 1, &h, 1, ch, 1);

	//generate a hue version of the testing image
	Mat comp_hue, comp_hue_output;
	cvtColor(compositeImage, comp_hue_output, CV_BGR2HSV);
	comp_hue.create(comp_hue_output.size(), comp_hue_output.depth());
	mixChannels(&comp_hue_output, 1, &comp_hue, 1, ch, 1);

	//generate histogram of the training data to backproject
	calcHist(&h, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	//backproject to combine probabilities of testing and training
	calcBackProject(&comp_hue, 1, 0, hist, backprojection, ranges, 1, true);
	threshold(backprojection, backprojection, 220, 255, THRESH_BINARY | CV_THRESH_OTSU);
	dilate(backprojection, backprojection, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//remove broken bits 
	Mat closed_src;
	Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(backprojection, closed_src, MORPH_CLOSE, structuringElement);

	Mat binary_inverted;
	threshold(closed_src, binary_inverted, 125.0, 255.0, THRESH_BINARY_INV);

	//bitwise against original image to remove the background
	Mat bitwise_mask;
	bitwise_and(compositeImage, compositeImage, bitwise_mask, binary_inverted);
	imshow("Binary inverted", binary_inverted);

	//outline the result
	Mat outline;
	Canny(binary_inverted, outline, 100, 255);
	//imshow("Outline", outline);

	Mat white, black;
	cvtColor(compositeImage, white, COLOR_HSV2BGR);
	threshold(compositeImage, white, 120, 255, CV_THRESH_BINARY);
	imshow("Orig", white);
}