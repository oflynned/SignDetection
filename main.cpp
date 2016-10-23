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

void segregate();
void separateGroundTruth();
void generateMetrics(string name, Mat& locations_found, Mat& ground_truth);

Mat black, white, red;
Mat comp_black;

int main(int argc, const char** argv) {
	segregate();
	separateGroundTruth();
	while (cv::waitKey(30) != ESC_KEY) {}
	return 0;
}

void separateGroundTruth() {
	Mat gt_red, gt_black, gt_white;

	//segregate ground truth into black only
	inRange(groundTruthImage, Scalar(0, 0, 0), Scalar(0, 0, 0), gt_black);
	//imshow("gtruth_b", gt_black);

	//segregate ground truth into white only
	inRange(groundTruthImage, Scalar(255, 255, 255), Scalar(255, 255, 255), gt_white);
	//imshow("gtruth_w", gt_white);

	//segregate ground truth into red only
	inRange(groundTruthImage, Scalar(0, 0, 255), Scalar(0, 0, 255), gt_red);
	//imshow("gtruth_r", gt_red);

	imshow("AYYYY BLACK", comp_black);
	imshow("GT BLACK", gt_black);

	generateMetrics("Red", red, gt_red);
	//generateMetrics("White", white, gt_white);
	//generateMetrics("Black", comp_black, gt_black);
}

void generateMetrics(string name, Mat& locations_found, Mat& ground_truth) {
	double precision, recall, accuracy, specificity, f1;
	int false_positives = 0;
	int false_negatives = 0;
	int true_positives = 0;
	int true_negatives = 0;
	for (int row = 0; row < ground_truth.rows; row++) {
		for (int col = 0; col < ground_truth.cols; col++) {
			uchar result = locations_found.at<uchar>(row, col);
			uchar gt = ground_truth.at<uchar>(row, col);
			if (gt > 0)
				if (result > 0)
					true_positives++;
				else false_negatives++;
			else if (result > 0)
				false_positives++;
			else true_negatives++;
		}
	}
	precision = ((double)true_positives) / ((double)(true_positives + false_positives));
	recall = ((double)true_positives) / ((double)(true_positives + false_negatives));
	accuracy = ((double)(true_positives + true_negatives)) / ((double)(true_positives + false_positives + true_negatives + false_negatives));
	specificity = ((double)true_negatives) / ((double)(false_positives + true_negatives));
	f1 = 2.0*precision*recall / (precision + recall);

	cout << name << ":" << endl << "Precision: " << precision << ", Recall: " << recall << ", Accuracy: " << accuracy << ", Specificity: " << specificity << ", f1: " << f1 << endl << endl;
}

void segregate() {
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
	threshold(backprojection, backprojection, 220, 255, THRESH_BINARY_INV);

	//remove broken bits 
	Mat closed_src;
	morphologyEx(backprojection, closed_src, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	imwrite("closed_src.png", closed_src);

	Mat temp_range_red;
	Mat src_copy = imread(composite, 1);
	cvtColor(src_copy, temp_range_red, CV_BGR2HLS);
	inRange(temp_range_red, Scalar(0, 0, 75), Scalar(180, 180, 180), temp_range_red);
	morphologyEx(temp_range_red, temp_range_red, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	imwrite("temp_range_red.png", temp_range_red);

	//bitwise against original image to remove the background
	Mat compositeImage_copy;
	compositeImage_copy = compositeImage;
	bitwise_and(compositeImage_copy, compositeImage_copy, red, temp_range_red);
	threshold(red, red, 0, 255, THRESH_BINARY);
	morphologyEx(red, red, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	//imshow("Red", red);

	/***p1 done***/

	//find inside shapes in red signs
	Mat temp_bw;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	//outline and fill in contours
	Canny(red, temp_bw, 100, 200);
	findContours(temp_bw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(temp_bw, contours, -1, Scalar(255), CV_FILLED);

	// create new image + set bg colour
	Mat crop(compositeImage.rows, compositeImage.cols, CV_8UC3);
	crop.setTo(Scalar(0, 255, 0));

	//copy back to original image to show only crops
	compositeImage.copyTo(crop, temp_bw);
	normalize(temp_bw.clone(), temp_bw, 0.0, 255.0, CV_MINMAX, CV_8UC1);
	//imshow("Contoured", temp_bw);

	//crop - bin_inv = b/w
	//get areas to remove from crop
	Mat bw(red.rows, red.cols, CV_8UC3);
	cvtColor(red, bw, CV_BGR2GRAY);
	threshold(bw, bw, 10, 255, THRESH_BINARY);

	//copy back to original image to show only crops
	Mat inner_areas(crop.rows, crop.cols, CV_8UC3);
	inner_areas.setTo(Scalar(0, 0, 0));
	crop.copyTo(inner_areas, bw);
	normalize(bw.clone(), bw, 0.0, 255.0, CV_MINMAX, CV_8UC1);
	cvtColor(inner_areas, inner_areas, CV_BGR2GRAY);
	threshold(inner_areas, inner_areas, 0, 255, THRESH_BINARY_INV);
	floodFill(inner_areas, Point(0, 0), Scalar(0, 255, 0));
	crop.copyTo(bw, inner_areas);

	//binary threshold on inner parts of sign
	floodFill(bw, Point(0, 0), Scalar(0, 255, 0));
	cvtColor(bw, bw, CV_BGR2GRAY);
	threshold(bw, black, 85, 255, THRESH_BINARY_INV);
	morphologyEx(black, black, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
	black.copyTo(comp_black);
	//imshow("COMP Black", comp_black);

	/***p2a done***/

	Mat black_overlay = black;
	threshold(black_overlay, black_overlay, 0, 255, THRESH_BINARY_INV);
	//imshow("overlay", black_overlay);

	//sign insides
	floodFill(bw, Point(0, 0), Scalar(0, 0, 0));
	threshold(bw, white, 0, 255, THRESH_OTSU);

	//for the love of fucking God work
	floodFill(black_overlay, Point(0, 0), Scalar(0, 0, 0));
	bitwise_or(white, black_overlay, white);
	//imshow("White", white);
	/***p2b done***/
}