// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <queue>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void changeGreyLevels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar newVal;
				if (val + 100 <= 255){
					newVal = val + 100;
				}
				else newVal = 255;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("greyed image", dst);
		waitKey();
	}
}
void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

int calcArea(Mat *src, Vec3b col) {
	int area = 0;
	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {
			if ((*src).at<Vec3b>(i, j) == col) {
				area++;
			}
		}
	}
	return area;
}

Point calcCenterOfMass(Mat *src, int area, Vec3b c) {
	Point center;
	int row = 0, col = 0;
	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {
			if ((*src).at<Vec3b>(i, j) == c) {
				row += i;
				col += j;
			}
		}
	}
	row /= area;
	col /= area;
	center = Point(col, row);
	return center;
}

double calcAxisOfElongation(Mat *src, Vec3b col, Point center) {
	int nom = 0;
	int den1 = 0, den2 = 0, denom = 0;
	double rad;
	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {
			if ((*src).at<Vec3b>(i, j) == col) {
				nom += (i - center.y) * (j - center.x);
				den1 += pow(j - center.x, 2);
				den2 += pow(i - center.y, 2);
			}
		}
	}
	nom *= 2;
	denom = den1 - den2;
	rad = atan2(nom, denom) / 2;
	if (rad < 0) {
		rad += CV_PI;
	}
	return rad;
}

int calcPerimeter(Mat *src, Vec3b col) {
	int NP = 0;
	for (int i = 1; i < (*src).rows - 1; i++) {
		for (int j = 1; j < (*src).cols - 1; j++) {
			if ((*src).at<Vec3b>(i, j) == col) {
				if ((*src).at<Vec3b>(i - 1, j - 1) != col ||
					(*src).at<Vec3b>(i - 1, j) != col ||
					(*src).at<Vec3b>(i - 1, j + 1) != col ||
					(*src).at<Vec3b>(i, j - 1) != col ||
					(*src).at<Vec3b>(i, j + 1) != col ||
					(*src).at<Vec3b>(i + 1, j - 1) != col ||
					(*src).at<Vec3b>(i + 1, j) != col ||
					(*src).at<Vec3b>(i + 1, j + 1) != col) 
				{
					NP++;
				}
			}
		}
	}
	return (NP * CV_PI) / 4;
}

double calcAspectRatio(Mat* src, Vec3b col) {
	int cmax = 0, cmin = (*src).cols;
	int rmax = 0, rmin = (*src).rows;
	for (int i = 1; i < (*src).rows - 1; i++) {
		for (int j = 1; j < (*src).cols - 1; j++) {
			if ((*src).at<Vec3b>(i, j) == col) {
					cmin = min(cmin, j);
					cmax = max(cmax, j);
					rmin = min(rmin, i);
					rmax = max(rmax, i);
			}
		}
	}
	return (cmax - cmin + 1) / ((double)rmax - rmin + 1);
}

Point getExtremes(Mat* src, Vec3b col) {
	int cmax = 0, cmin = (*src).cols;
	int rmax = 0, rmin = (*src).rows;
	for (int i = 1; i < (*src).rows - 1; i++) {
		for (int j = 1; j < (*src).cols - 1; j++) {
			if ((*src).at<Vec3b>(i, j) == col) {
				cmin = min(cmin, j);
				cmax = max(cmax, j);
				rmin = min(rmin, i);
				rmax = max(rmax, i);
			}
		}
	}
	Point ex = Point(cmin, cmax);
	return ex;
}
void drawAxisElong(Mat* src, Vec3b col, Point center, double rad) {
	Point extremes = getExtremes(src, col);
	int ra = center.y + tan(rad) * (extremes.x - center.x);
	int rb = center.y + tan(rad) * (extremes.y - center.x);
	Point a(extremes.x, ra);
	Point b(extremes.y, rb);
	Mat dst = (*src).clone();
	line(dst, a, b, Scalar(0, 0, 0), 2);
	imshow("Axis of elongation", dst);
}

void drawProjectionH(Mat *src, Vec3b col) {
	Mat dst = Mat((*src).rows, (*src).cols, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 1; i < (*src).rows - 1; i++) {
		Point a = Point(0, i);
		Point b = Point(0, i);
		for (int j = 1; j < (*src).cols - 1; j++) {
			if ((*src).at<Vec3b>(i, j) == col) {
				b.x = b.x + 1;
			}
		}
		line(dst, a, b, Scalar(255, 255, 0), 1);
	}
	imshow("Horizontal projection", dst);
}

void drawProjectionV(Mat *src, Vec3b col) {
	Mat dst = Mat((*src).rows, (*src).cols, CV_8UC3, Scalar(0, 0, 0));
	for (int j = 1; j < (*src).cols - 1; j++) {
		Point a = Point(j, 0);
		Point b = Point(j, 0);
		for (int i = 1; i < (*src).rows - 1; i++) {
			if ((*src).at<Vec3b>(i, j) == col) {
				b.y = b.y + 1;
			}
		}
		line(dst, a, b, Scalar(0, 255, 255), 1);
	}
	imshow("Vertical projection", dst);
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	uchar r, g, b;
	if (event == CV_EVENT_LBUTTONDBLCLK)
		{
			r = (uchar)(*src).at<Vec3b>(y, x)[2];
			g = (uchar)(*src).at<Vec3b>(y, x)[1];
			b = (uchar)(*src).at<Vec3b>(y, x)[0];
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				r,
				g,
				b);
			Vec3b color = Vec3b(b, g, r);
			int area = calcArea(src, color);
			std::cout << "Area of object:\t\t\t\t" << area << std::endl;
			Point centerOfMass = calcCenterOfMass(src, area, color);
			std::cout << "Center of mass:\t\t\t\t" << centerOfMass.y << " row  ,  " << centerOfMass.x << " col" << std::endl;
			double rad = calcAxisOfElongation(src, color, centerOfMass);
			int axis = rad * 180 / CV_PI;
			std::cout << "Axis of elongation (deg):\t\t" << axis << std::endl;
			int perim = calcPerimeter(src, color);
			std::cout << "Perimeter (rad):\t\t\t" << perim << std::endl;
			double thinnesRatio = 4 * CV_PI * (area / (double)(perim * perim));
			std::cout << "Thinnes ratio:\t\t\t\t" << thinnesRatio << std::endl;
			double aspectRatio = calcAspectRatio(src, color);
			std::cout << "Aspect ratio:\t\t\t\t" << aspectRatio << std::endl;
			drawAxisElong(src, color, centerOfMass, rad);
			drawProjectionH(src, color);
			drawProjectionV(src, color);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void geometrical_features()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}
/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void fourSquares(){
	Mat_<Vec3b> img(600, 800, Vec3b(255, 255, 255));
	int width = 800;
	int height = 600;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			if (j > width / 2){
				if (i < height / 2){
					img(i, j) = Vec3b(0, 0, 255);
				}
				else{
					img(i, j) = Vec3b(0, 255, 255);
				}
			}
			else{
				if (i >= height / 2){
					img(i, j) = Vec3b(0, 255, 0);
				}
			}
		}
	imshow("four squares", img);
	waitKey();
}

void inverseMatrix(){
	float vals[9] = { 3, 2, 3, 4, 5, 6, 7, 8, 9 };
	Mat M(3, 3, CV_32FC1, vals); //4 parameter constructor
	std::cout << M.inv() << std::endl;
	getchar();
	getchar();
}

void splitRGB(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat_<Vec3b> src;
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int width = src.cols;
		int height = src.rows;
		Mat_<Vec3b> redImg(height, width, Vec3b(255, 255, 255));
		Mat_<Vec3b> greenImg(height, width, Vec3b(255, 255, 255));
		Mat_<Vec3b> blueImg(height, width, Vec3b(255, 255, 255));
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				Vec3b pixel = src(i, j);
				uchar b = pixel[0], g = pixel[1], r = pixel[2];
				redImg(i, j) = Vec3b(0, 0, r);
				blueImg(i, j) = Vec3b(b, 0, 0);
				greenImg(i, j) = Vec3b(0, g, 0);
			}
		}
		imshow("Source", src);
		imshow("Red", redImg);
		imshow("Green", greenImg);
		imshow("Blue", blueImg);
		waitKey();
	}
}

void colorToGrayscale(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat_<Vec3b> src;
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int width = src.cols;
		int height = src.rows;
		Mat_<uchar> grayscale(height, width, (uchar)255);
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				Vec3b p = src(i, j);
				uchar b = p[0], g = p[1], r = p[2];
				grayscale(i, j) = (b + g + r) / 3;
			}
		}
		imshow("Source", src);
		imshow("Grayscale", grayscale);
		waitKey();
	}
}

void grayscaleToBW(uchar threshold){
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat_<uchar> src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		Mat_<uchar> bw(height, width, (uchar)255);
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				uchar p = src(i, j);
				if (p < threshold){
					bw(i, j) = 0;
				}
				else bw(i, j) = 255;
			}
		}
		imshow("Source", src);
		imshow("Black/white", bw);
		waitKey();
	}
}

void rgbToHsv(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat_<Vec3b> src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int width = src.cols;
		int height = src.rows;
		Mat_<uchar> Hue(height, width, (uchar)255);
		Mat_<uchar> Sat(height, width, (uchar)255);
		Mat_<uchar> Val(height, width, (uchar)255);
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				Vec3b p = src(i, j);
				float B = p[0], G = p[1], R = p[2];
				float r = R / 255, g = G / 255, b = B / 255;
				float M = max(r, g, b);
				float m = min(r, g, b);
				float C = M - m;
				float V = M;
				float S;
				if (V != 0){
					S = C / V;
				}
				else S = 0;
				float H;
				if (C != 0){
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else H = 0;
				if (H < 0)
					H = H + 360;
				Hue(i, j) = H*255/360;
				Sat(i, j) = S*255;
				Val(i, j) = V*255;
			}
		}
		imshow("Source", src);
		imshow("Hue", Hue);
		imshow("Saturation", Sat);
		imshow("Value", Val);
		waitKey();
	}
}

Mat_<uchar> intensityHistogram(float *normalHisto, bool show) {
	char fname[MAX_PATH];
	int histo[256] = { 0 };
	int M;
	Mat_<uchar> src;
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		M = src.rows * src.cols;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar pixel = src(i, j);
				histo[pixel]++;
			}
		}
		for (int i = 0; i < 256; i++) {
			normalHisto[i] = (float)histo[i] / M;
		}
		if (show) {
			imshow("Source", src);
			showHistogram("Intensity of gray levels", histo, 256, 256);
			waitKey();
		}
		else break;
	}
	return src;
}

Mat_<uchar> multilevelThresholding(float *normalHisto, Mat_<uchar> src, int* maxList, bool show) {
	char fname[MAX_PATH];
	int maxSize = 1;
	int WH = 5;
	float TH = 0.0003;
	for (int k = 0 + WH; k < 256 - WH; k++) {
		float v = 0;
		float maxVal = 0;
		for (int j = k - WH; j <= k + WH; j++) {
			v += normalHisto[j];
			if (normalHisto[j] > maxVal)
				maxVal = normalHisto[j];
		}
		v /= 2 * WH + 1;
		if (normalHisto[k] > v + TH && normalHisto[k] >= maxVal) {
			maxList[maxSize] = k;
			maxSize++;
		}
	}
	maxList[maxSize] = 255;
	maxSize++;
	Mat_ <uchar> dst = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar p = src(i, j);
			int k = 0;
			while (maxList[k + 1] < p)
				k++;
			if ((p - maxList[k]) > (maxList[k + 1] - p)) {
				dst(i, j) = maxList[k + 1];
			}
			else {
				dst(i, j) = maxList[k];
			}
		}
	}
	if (show) {
		imshow("Source", src);
		imshow("Multilevel thresholding", dst);
		waitKey();
	}
	return dst;
}

double clip(double n,double lower, double upper) {
	return max(lower, min(n, upper));
}

void floydSteinbergDithering(Mat_ <uchar> src, int* maxList){
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar oldpixel = src(i, j);
			uchar newpixel;

			int k = 0;
			while (maxList[k + 1] < oldpixel)
				k++;
			if ((oldpixel - maxList[k]) > (maxList[k + 1] - oldpixel)) {
				newpixel = maxList[k + 1];
			}
			else {
				newpixel = maxList[k];
			}

			src(i, j) = newpixel;
			int err = oldpixel - newpixel;
			if (j + 1 < src.cols) {
				src(i, j + 1) = clip(src(i, j + 1) + 7 * err / 16, 0, 255);
			}
			if (i + 1 < src.rows) {
				if (j - 1 >= 0) {
					src(i + 1, j - 1) = clip(src(i + 1, j - 1) + 3 * err / 16, 0, 255);
				}
				src(i + 1, j) = clip(src(i + 1, j) + 5 * err / 16, 0, 255);
				if (j + 1 < src.cols) {
					src(i + 1, j + 1) = clip(src(i + 1, j + 1) + err / 16, 0, 255);
				}
			}
		}
	}
	imshow("Dithering", src);
	waitKey();
}

void colorLabeledImg(const Mat_<int> &labels, int no) {
	uchar *reds = (uchar*)malloc(sizeof(uchar) * no);
	uchar *greens = (uchar*)malloc(sizeof(uchar) * no);
	uchar *blues = (uchar*)malloc(sizeof(uchar) * no);
	Mat dst(labels.rows, labels.cols, CV_8UC3, Scalar(255, 255, 255));
	srand(time(NULL));

	for (int i = 0; i < no; i++) {
		reds[i] = rand() % 256;
		greens[i] = rand() % 256;
		blues[i] = rand() % 256;
	}
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int val = labels(i, j);
			if (val)
				dst.at<Vec3b>(i, j) = Vec3b(blues[val], reds[val], greens[val]);
		}
	}
	free(reds);
	free(greens);
	free(blues);
	imshow("Image", dst);
}

bool pointInside(int i, int j, int rows, int cols)
{
	if (i < 0 || i >= rows) return false;
	if (j < 0 || j >= cols) return false;
	return true;
}

void labelBF() {
	char fname[MAX_PATH];
	Mat src;
	while (openFileDlg(fname)) {
		int label = 0;
		src = imread(fname, IMREAD_GRAYSCALE);
		Mat labels = Mat::zeros(src.rows, src.cols, CV_32SC1);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					label++;
					std::queue<Point> Q;
					labels.at<int>(i, j) = label;
					Q.push(Point(j, i));
					while (!Q.empty()) {
						Point p = Q.front();
						Q.pop();
						int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
						int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
						for (int n = 0; n < 8; n++) {
							int x = p.x + dy[n];
							int y = p.y + dx[n];
							if (pointInside(x, y, src.cols, src.rows) && src.at<uchar>(y, x) == 0 && labels.at<int>(y, x) == 0) {
								labels.at<int>(y, x) = label;
								Q.push(Point(x, y));
							}
						}
					}
				}
			}
		}
		colorLabeledImg(labels, label);
	}
}

void labelTwoPass()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		int label = 0;
		int height = src.rows;
		int width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);

		int dy[4] = { -1, -1, -1, 0 };
		int dx[4] = { -1, 0, 1, -1 };

		std::vector<std::vector<int>> edges;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					std::vector<int> L;

					for (int k = 0; k < 4; k++)
					{
						int y = j + dx[k];
						int x = i + dy[k];

						if (pointInside(x, y, height, width) && labels.at<int>(x, y) > 0)
						{
							L.push_back(labels.at<int>(x, y));
						}
					}

					if (L.size() == 0)
					{
						label++;
						edges.resize(label + 1);
						labels.at<int>(i, j) = label;
					}

					else
					{
						int x = *std::min_element(L.begin(), L.end());
						labels.at<int>(i, j) = x;

						for (int n = 0; n < L.size(); n++)
						{
							int y = L[n];

							if (y != x)
							{
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}
					}
				}
			}
		}

		int newLabel = 0;
		int* newLabels = (int*)malloc(sizeof(int) * (label + 1));

		for (int i = 0; i < label + 1; i++)
		{
			newLabels[i] = 0;
		}

		for (int i = 1; i < label + 1; i++)
		{
			if (newLabels[i] == 0)
			{
				newLabel++;

				std::queue<int> Q;
				newLabels[i] = newLabel;
				Q.push(i);

				while (!Q.empty())
				{
					int x = Q.front();
					Q.pop();

					for (int k = 0; k < edges[x].size(); k++)
					{
						int y = edges[x].at(k);

						if (newLabels[y] == 0)
						{
							newLabels[y] = newLabel;
							Q.push(y);
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
			}
		}
		colorLabeledImg(labels, newLabel);
	}
}

void startTracing(Mat_<uchar> src, int i, int j) {
	Point P0(i, j);
	Point P1;
	Point Pn, Pn_1;
	int dir = 7, n = 0;
	int dir_i[] = {0, -1, -1, -1, 0, 1, 1, 1};
	int dir_j[] = {1, 1, 0, -1, -1, -1, 0, 1};
	Mat_<uchar> dst(src.rows, src.cols, (uchar)255);
	std::vector<int> AC;
	std::vector<int> DC;
	dst(i, j) = 0;
	std::cout << "P0:	" << i << " " << j << std::endl;
	do {
		if (dir % 2 == 0)
			dir = (dir + 7) % 8;
		else
			dir = (dir + 6) % 8;
		while (src(i + dir_i[dir], j + dir_j[dir]) != 0) {
			dir = (dir + 1) % 8;
		}
		AC.push_back(dir);
		Pn_1 = Point(i, j);
		i += dir_i[dir];
		j += dir_j[dir];
		Pn = Point(i, j);
		dst(i, j) = 0;
		n++;
		if (n == 1) {
			P1 = Point(i, j);
			std::cout << "P1:	" << i << " " << j << std::endl;
		}
	} while ((n < 2) || (Pn != P1) || (Pn_1 != P0));
	imshow("Object", src);
	imshow("Contour", dst);
	std::cout << "AC: ";
	for (int i = 0; i < AC.size(); i++) {
		int first = AC.at(i);
		std::cout << first << " ";
		int second;
		if (i + 1 < AC.size())
			second = AC.at(i + 1);
		else
			second = AC.at(0);
		if (first <= second)
			DC.push_back(second - first);
		else
			DC.push_back(8 - first + second);
	}
	std::cout << std::endl << "DC: ";
	for (int i = 0; i < DC.size(); i++) {
		std::cout << DC.at(i) << " ";
	}
	std::cout << std::endl;
}

void borderTracing() {
	Mat_<uchar> src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		//getting the first black pixels position
		int i = 0, j;
		bool found = false;
		for (; i < src.rows && !found; i++) {
			j = 0;
			for (; j < src.cols && !found; j++) {
				if (src(i, j) == 0) {
					found = true;
				}
			}
		}
		startTracing(src, i - 1, j - 1);
	}
}

void contourReconstruction() {
	std::ifstream file;
	file.open("Images/border_tracing/reconstruct.txt");
	int i, j, n, dir;
	file >> i >> j;
	file >> n;
	Mat_<uchar> src = imread("Images/border_tracing/gray_background.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int dir_i[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dir_j[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	while (file >> dir) {
		src(i, j) = 0;
		i += dir_i[dir];
		j += dir_j[dir];
	}
	imshow("Contour", src);
	waitKey();
}

void getWeights(double x, double w[]) {
	const float A = -0.75f;
	w[0] = ((A*(x + 1) - 5 * A)*(x + 1) + 8 * A)*(x + 1) - 4 * A;
	w[1] = ((A + 2)*x - (A + 3))*x*x + 1;
	w[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
	w[3] = 1.0f - w[0] - w[1] - w[2];
}
double cubicInterpolate(double p[4], double x) {
	return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])));
}

double bicubicInterpolate(double p[4][4], double x, double y) {
	double arr[4];
	arr[0] = cubicInterpolate(p[0], y);
	arr[1] = cubicInterpolate(p[1], y);
	arr[2] = cubicInterpolate(p[2], y);
	arr[3] = cubicInterpolate(p[3], y);
	return cubicInterpolate(arr, x);
}
void zoomImage() {
	double scaleW, scaleH;
	std::cout << "Horizontal scaling:" << std::endl;
	std::cin >> scaleW;
	std::cout << "Vertical scaling:" << std::endl;
	std::cin >> scaleH;
	std::cout << scaleH << " " << scaleW;
	Mat_<uchar> src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Source", src);
		//Nearest neighbor
		Mat_<uchar> dst(src.rows * scaleH, src.cols * scaleW, 255);
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				dst(i, j) = src(i / scaleH, j / scaleW);
			}
		}
		imshow("Nearest", dst);

		//Bilinear
		Mat_ <uchar> test(2, 2, 10);
		test(0, 1) = 20;
		test(1, 0) = 30;
		test(1, 1) = 40;
		Mat_<uchar> dst2(src.rows * scaleH, src.cols * scaleW, 255);
		for (int i = 0; i < dst2.rows; i++) {
			for (int j = 0; j < dst2.cols; j++) {
				double srcI = (double)i / scaleH;
				double srcJ = (double)j / scaleW;
				double wx = srcJ - floor(srcJ);
				double wy = srcI - floor(srcI);
				int fI = floor(srcI);
				int fJ = floor(srcJ);
				int sI = fI, sJ = fJ;
				if (fI + 1 < src.rows) {
					sI = fI + 1;
				}
				if (fJ + 1 < src.cols) {
					sJ = fJ + 1;
				}
				int p1 = src(fI, fJ);
				int p2 = src(fI, sJ);
				int p3 = src(sI, fJ);
				int p4 = src(sI, sJ);
				int interpX1 = (1 - wx) * p1 + wx * p2;
				int interpX2 = (1 - wx) * p3 + wx * p4;
				int interpY = (1 - wy) * interpX1 + wy * interpX2;
				dst2(i, j) = interpY;
			}
		}
		imshow("Bilinear", dst2);

		//Bicubic
		Mat_<uchar> dst3(src.rows * scaleH, src.cols * scaleW, (uchar)0);

		Mat_<uchar> src2(src.rows, src.cols, (uchar)0);
		copyMakeBorder(src, src2, scaleH, scaleH, scaleW, scaleW, BORDER_REPLICATE);
		for (int i = 0; i < dst3.rows; i++) {
			for (int j = 0; j < dst3.cols; j++) {
				//Calculate 4x4 neighborhood in source image
				double x = (j / scaleW) - cvFloor(j / scaleW);
				double y = (i / scaleH) - cvFloor(i / scaleH);
				int srcI = floor(i / scaleH);
				int srcJ = floor(j / scaleW);

				int startI = srcI;
				int endI = startI + 3;
				int startJ = srcJ;
				int endJ = startJ + 3;
				double p[4][4] = { 0.0f };
				for (int _i = 0; _i < 4; _i++) {
					for (int _j = 0; _j < 4; _j++) {
						p[_i][_j] = src2(srcI + _i, srcJ + _j);
					}
				}
				dst3(i, j) = clip(bicubicInterpolate(p, y, x), 0, 255);

			}
		}
		imshow("Bicubic", dst3);
	}
}

Mat_<uchar> dilation(Mat_<uchar> src, int nr) {
	int dx[] = { 0, 0, -1, 1 };
	int dy[] = { -1, 1, 0, 0 };
	Mat_<uchar> prev_dst = src.clone();
	Mat_<uchar> dst = src.clone();
	for (int k = 0; k < nr; k++) {
		for (int i = 0; i < prev_dst.rows; i++) {
			for (int j = 0; j < prev_dst.cols; j++) {
				if (prev_dst(i, j) == 0)
					for (int d = 0; d < 4; d++) {
						if (pointInside(i + dy[d], j + dx[d], dst.rows, dst.cols))
							dst(i + dy[d], j + dx[d]) = 0;
					}
			}
		}
		prev_dst = dst.clone();
	}
	return dst;
}

void dilate(int nr) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Original", src);
		Mat_<uchar> dst = dilation(src, nr);
		imshow("Dilation", dst);
	}
}

Mat_<uchar> erosion(Mat_<uchar> src, int nr) {
	int dx[] = { 0, 0, -1, 1 };
	int dy[] = { -1, 1, 0, 0 };
	Mat_<uchar> prev_dst = src.clone();
	Mat_<uchar> dst = src.clone();
	for (int k = 0; k < nr; k++) {
		for (int i = 0; i < prev_dst.rows; i++) {
			for (int j = 0; j < prev_dst.cols; j++) {
				if (prev_dst(i, j) == 0) {
					boolean erode = false;
					for (int d = 0; d < 4; d++) {
						if (pointInside(i + dy[d], j + dx[d], prev_dst.rows, prev_dst.cols))
							if (prev_dst(i + dy[d], j + dx[d]) != 0)
								erode = true;
					}
					if (erode)
						dst(i, j) = 255;
				}
			}
		}
		prev_dst = dst.clone();
	}
	return dst;
}

void erode(int nr) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Original", src);
		Mat_<uchar> dst = erosion(src, nr);
		imshow("Erosion", dst);
	}
}

void opening() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> dst = erosion(src, 1);
		dst = dilation(dst, 1);
		imshow("Original", src);
		imshow("Opening", dst);
	}
}

void closing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> dst = dilation(src, 1);
		dst = erosion(dst, 1);
		imshow("Original", src);
		imshow("Closing", dst);
	}
}

void extractBoundary() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> eroded = erosion(src, 1);
		Mat_<uchar> boundary(src.rows, src.cols, 255);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src(i, j) == 0 && eroded(i, j) == 255) {
					boundary(i, j) = 0;
				}
			}
		}
		imshow("Original", src);
		imshow("Boundary", boundary);
	}
}

Mat_<uchar> complement(Mat_<uchar> src) {
	Mat_<uchar> dst(src.rows, src.cols, 255);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == 255)
				dst(i, j) = 0;
		}
	}
	return dst;
}

bool isEqual(Mat_<uchar> m1, Mat_<uchar> m2) {
	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			if (m1(i, j) != m2(i, j)) {
				return false;
			}
		}
	}
	return true;
}
void fillArea() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> Xk_1(src.rows, src.cols, 255);
		Mat_<uchar> comp = complement(src);
		imshow("Source", src);
		imshow("Complement", comp);
		Xk_1(Xk_1.rows / 2, Xk_1.cols / 2) = 0;
		Mat_<uchar> Xk = Xk_1.clone();
		do {
			Xk.copyTo(Xk_1);
			Mat_<uchar> dilated = dilation(Xk_1, 1);
			for (int i = 0; i < Xk.rows; i++) {
				for (int j = 0; j < Xk.cols; j++) {
					if (dilated(i, j) == comp(i, j))
						Xk(i, j) = dilated(i, j);
					else
						Xk(i, j) = 255;
				}
			}
		} while (!isEqual(Xk, Xk_1));
		imshow("Filled", Xk);
	}
}

void calcHisto(Mat_<uchar> src, int *histo) {
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			histo[src(i, j)]++;
		}
	}
}

void statisticalProperties() {
	char fname[MAX_PATH];
	double pdf[256];
	int M;
	Mat_<uchar> src;
	while (openFileDlg(fname)) {
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		M = src.rows * src.cols;
		int histo[256] = { 0 };
		int iMin = -1, iMax = -1;
		calcHisto(src, histo);
		double mean = 0;
		double stdDev = 0.0f;
		for (int g = 0; g < 256; g++) {
			mean += g * histo[g];
			pdf[g] = (double)histo[g] / M;
			if (histo[g] != 0) {
				if (iMin == -1) {
					iMin = g;
				}
				iMax = g;
			}
		}
		mean /= M;
		for (int g = 0; g < 256; g++) {
			stdDev += (double)pow(g - (int)mean, 2) * pdf[g];
		}
		stdDev = sqrt(stdDev);
		imshow("Source", src);
		showHistogram("Histogram", histo, 256, 256);
		std::cout << "Mean:\t\t\t" << mean << std::endl;
		std::cout << "Standard deviation\t" << stdDev << std::endl;
		waitKey();
		//Global thresholding
		int Tk_1 = (iMax + iMin)/2;
		int epsilon = 1;
		int Tk = -1;
		do {
			if (Tk != -1)
				Tk_1 = Tk;
			int m1 = 0;
			int m2 = 0;
			int n1 = 0, n2 = 0;
			for (int g = iMin; g < Tk_1; g++) {
				m1 += g * histo[g];
				n1 +=  histo[g];
			}
			m1 = m1 / n1;
			for (int g = Tk_1; g < iMax; g++) {
				m2 += g * histo[g];
				n2 += histo[g];
			}
			m1 = m1 / n1;
			m2 = m2 / n2;
			Tk = (m1 + m2) / 2;
		} while (abs(Tk - Tk_1) < epsilon);
		std::cout << "Error:" << std::endl;
		std::cin >> epsilon;
		std::cout << "Threshold\t\t" << Tk << std::endl;
		Mat_<uchar> dst(src.rows, src.cols, (uchar)0);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src(i, j) >= Tk)
					dst(i, j) = 255;
			}
		}
		imshow("Source", src);
		imshow("Thresholded", dst);
		waitKey();

		int offset;
		std::cout << "Brightness offset\t" << std::endl;
		std::cin >> offset;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
					dst(i, j) = clip(src(i, j) + offset, 0, 255);
			}
		}

		imshow("Source", src);
		imshow("Brightness change", dst);
		int histo2[256] = { 0 };
		calcHisto(dst, histo2);
		showHistogram("Old", histo, 256, 256);
		showHistogram("New", histo2, 256, 256 + offset);
		waitKey();

		int gInMin = iMin, gInMax = iMax;
		int gOutMin, gOutMax;
		std::cout << "gOutMin, gOutMax:\n";
		std::cin >> gOutMin >> gOutMax;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst(i, j) = gOutMin + (src(i, j) - iMin) *  (gOutMax - gOutMin) / (iMax - iMin);
			}
		}
		imshow("Source", src);
		imshow("Contrast change", dst);
		int histo3[256] = { 0 };
		calcHisto(dst, histo3);
		showHistogram("Old", histo, 256, 256);
		showHistogram("New", histo3, 256, 256);
		waitKey();

		double gamma;
		std::cout << "Gamma:\n";
		std::cin >> gamma;
		std::cout << gamma << std::endl;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				double px = src(i, j);
				px = px / 255;
				px = powf(px, gamma);
				px = px * 255;
				dst(i, j) = clip((int)px, 0, 255);
			}
		}
		imshow("Source", src);
		imshow("Gamma correction", dst);
		int histo4[256] = { 0 };
		calcHisto(dst, histo4);
		waitKey();

		double cpdf[256] = { 0 };
		for (int i = 0; i < 256; i++) {
			for (int j = 0; j <= i; j++) {
				cpdf[i] += pdf[j];
			}
		}
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst(i, j) = 255 * cpdf[src(i, j)];
			}
		}
		imshow("Source", src);
		imshow("Equalized", dst);
		int histo5[256] = { 0 };
		calcHisto(dst, histo5);
		showHistogram("Old", histo, 256, 256);
		showHistogram("New", histo5, 256, 256);
		waitKey();
	}
}

void convolute(Mat_<uchar> src, Mat_<uchar> dst, int k, Mat_<double> kernel, int norm, double sum) {
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double val = 0;
			int pInside = 0;
			for (int x = -k/2; x <= k/2; x++) {
				for (int y = -k/2; y <= k/2; y++) {
					if (pointInside(i + y, j + x, src.rows, src.cols)) {
						pInside++;
						val += kernel(y + k / 2, x + k / 2) * src(i + y, j + x);
					}
				}
			}
			if (norm == 0) {//mean filter
				val = val / pInside;
			}
			else if (norm == 1) {//3x3 gaussian blur
				val = val / 16;
			}
			else if (norm == 3) {//saturation
				val = clip(val, 0, 255);
			}
			else if (norm == 4) {
				val /= sum;
			}
			dst(i, j) = val;
		}
	}
}

void convoluteDouble(Mat_<uchar> src, Mat_<double> dst, int k, Mat_<double> kernel) {
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double val = 0.0f;
			for (int x = -k / 2; x <= k / 2; x++) {
				for (int y = -k / 2; y <= k / 2; y++) {
					if (pointInside(i + y, j + x, src.rows, src.cols)) {
						val += kernel(y + k / 2, x + k / 2) * (double)src(i + y, j + x);
					}
				}
			}
			dst(i, j) = val;
		}
	}
}

void meanFilters() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols, (uchar)0);
		Mat_<uchar> dst2(src.rows, src.cols, (uchar)0);
		Mat_<uchar> dst3(src.rows, src.cols, (uchar)0);
		Mat_<uchar> dst4(src.rows, src.cols, (uchar)0);
		Mat_<uchar> dst5(src.rows, src.cols, (uchar)0);
		
		Mat_<int> k3(3, 3, (int)1);
		convolute(src, dst, 3, k3, 0, 0);
		imshow("Source", src);
		imshow("Mean 3x3", dst);

		Mat_<int> k5(5, 5, (int)1);
		convolute(src, dst2, 5, k5, 0, 0);
		imshow("Mean 5x5", dst2);

		Mat_<int> gauss3(3, 3, (int)1);
		gauss3(0, 1) = 2;
		gauss3(1, 0) = 2;
		gauss3(1, 2) = 2;
		gauss3(2, 1) = 2;
		gauss3(1, 1) = 4;
		convolute(src, dst3, 3, gauss3, 1, 0);
		imshow("Gauss 3x3", dst3);

		Mat_<int> laplace3(3, 3, (int)-1);
		laplace3(1, 1) = 8;
		convolute(src, dst4, 3, laplace3, 3, 0);
		imshow("Laplace 3x3", dst4);

		Mat_<int> highpass3(3, 3, (int)-1);
		highpass3(1, 1) = 8;
		convolute(src, dst5, 3, highpass3, 3, 0);
		imshow("High Pass 3x3", dst5);
	}
}

void centering_transform(Mat img) 
{
	//expects floating point image
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		} 
	} 
}

void frequency_domain(Mat src) {
	//convert input image to float image
	Mat srcf; src.convertTo(srcf, CV_32FC1);

	//centering transformation  
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels 
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	//display the phase and magnitude images here
	Mat_<uchar> logMag(src.rows, src.cols, (uchar)0);
	int maxlog = 0;
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			float val = mag.at<float>(i, j);
			if (log(val) > maxlog)
				maxlog = log(val);
		}
	}
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			logMag(i, j) = log(abs(mag.at<float>(i,j)) + 1) / maxlog * 255;
		}
	}
	imshow("Centered F spectrum", logMag);

	//insert filtering operations on Fourier coefficients here
	Mat channels2[] = { channels[0].clone(), channels[1].clone() };
	Mat channels3[] = { channels[0].clone(), channels[1].clone() };
	Mat channels4[] = { channels[0].clone(), channels[1].clone() };

	//LPF
	int R = 20;
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			float eq = pow(src.rows / 2 - i, 2) + pow(src.cols / 2 - j, 2);
			if (eq > R*R) {
				channels[0].at<float>(i, j) = 0.0f;
				channels[1].at<float>(i, j) = 0.0f;
			}
		}
	}

	//HPF
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			float eq = pow(src.rows / 2 - i, 2) + pow(src.cols / 2 - j, 2);
			if (eq <= R*R) {
				channels2[0].at<float>(i, j) = 0.0f;
				channels2[1].at<float>(i, j) = 0.0f;
			}
		}
	}

	int A = 10;
	//Gaussian LPF
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			float eq = (pow(src.rows / 2 - i, 2) + pow(src.cols / 2 - j, 2)) / (A*A);
			channels3[0].at<float>(i, j) = channels3[0].at<float>(i, j) * exp(-eq);
			channels3[1].at<float>(i, j) = channels3[1].at<float>(i, j) * exp(-eq);
		}
	}

	//Gaussian HPF
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			float eq = (pow(src.rows / 2 - i, 2) + pow(src.cols / 2 - j, 2)) / (A*A);
			channels4[0].at<float>(i, j) = channels4[0].at<float>(i, j) * (1 - exp(-eq));
			channels4[1].at<float>(i, j) = channels4[1].at<float>(i, j) * (1 - exp(-eq));
		}
	}
	//store in real part in channels[0] and imaginary part in channels[1]
	// ...... 

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	Mat dst2, dstf2, fourier2;
	Mat dst3, dstf3, fourier3;
	Mat dst4, dstf4, fourier4;

	merge(channels, 2, fourier);
	merge(channels2, 2, fourier2);
	merge(channels3, 2, fourier3);
	merge(channels4, 2, fourier4);

	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	dft(fourier2, dstf2, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	dft(fourier3, dstf3, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	dft(fourier4, dstf4, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transformation 
	centering_transform(dstf);
	centering_transform(dstf2);
	centering_transform(dstf3);
	centering_transform(dstf4);
	//normalize the result and put in the destination image

	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(dstf2, dst2, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(dstf3, dst3, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(dstf4, dst4, 0, 255, NORM_MINMAX, CV_8UC1);

	imshow("LPT", dst);
	imshow("HPT", dst2);
	imshow("Gaussian LPT", dst3);
	imshow("Gaussian HPT", dst4);
	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255]. 
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);
}

void fourier() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Src", src);
		frequency_domain(src);
	}
}

void medianFilter(int k) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols, (uchar)0);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				std::vector<int> vals;
				for (int x = -k / 2; x <= k / 2; x++) {
					for (int y = -k / 2; y <= k / 2; y++) {
						if (pointInside(i + y, j + x, src.rows, src.cols))
							vals.push_back(src(i + y, j + x));
					}
				}
				std::sort(vals.begin(), vals.end());
				dst(i, j) = vals[vals.size() / 2];
			}
		}
		imshow("Source", src);
		imshow("Median filter", dst);
	}
}

int closestOdd(float val) {
	int dec = val;
	if (dec % 2 == 0)
		dec++;
	return dec;
}

void gaussian2DFilter(float sigma) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount();
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols, (uchar)0);
		int w = closestOdd(6 * sigma);
		int x0 = w / 2, y0 = w / 2;
		printf("W=%d\n", w);
		Mat_<double> kernel(w, w, (double)0.0f);
		double sum = 0.0f;
		for (int x = -w / 2; x <= w / 2; x++) {
			for (int y = -w / 2; y <= w / 2; y++) {
				int xx = x + w / 2;
				int yy = y + w / 2;
				double val = (1 / (2 * CV_PI * sigma * sigma)) * exp(-((pow(x, 2) + pow(y, 2)) / (2 * sigma*sigma)));
				kernel(yy, xx) = val;
				sum += val;
			}
		}
		convolute(src, dst, w, kernel, 4, sum);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("Source", src);
		imshow("2D Gauss", dst);
	}
}

void gaussian2x1DFilter(float sigma) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount();
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> tmp(src.rows, src.cols, (uchar)0);
		Mat_<uchar> dst(src.rows, src.cols, (uchar)0);
		int w = closestOdd(6 * sigma);
		
		double *kernel = (double*)malloc(sizeof(double) * w);
		double sum = 0.0f;

		for (int i = 0; i < w; i++) {
			kernel[i] = (1 / (2 * CV_PI * sigma * sigma)) * exp(-(pow(i - w/2, 2) / (2*sigma*sigma)));
			sum += kernel[i];
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int val = 0;
				for (int x = -w / 2; x < w / 2; x++) {
					if (pointInside(i, j+x, src.rows, src.cols))
						val += kernel[x + w / 2] * src(i, j + x);
				}
				tmp(i, j) = val / sum;
			}
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int val = 0;
				for (int y = -w / 2; y < w / 2; y++) {
					if (pointInside(i+y, j, src.rows, src.cols))
						val += kernel[y + w / 2] * tmp(i + y, j);
				}
				dst(i, j) = val / sum;
			}
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("Source", src);
		imshow("Temp", tmp);
		imshow("Filter", dst);
	}
}
void canny() {
	char fname[MAX_PATH];
	float sigma = 0.5f;
	while (openFileDlg(fname)) {

		//Gaussian filtering
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> srcF(src.rows, src.cols, (uchar)0);
		int w = closestOdd(6 * sigma);
		int x0 = w / 2, y0 = w / 2;
		printf("W=%d\n", w);
		Mat_<double> kernel(w, w, (double)0.0f);
		double sum = 0.0f;
		for (int x = -w / 2; x <= w / 2; x++) {
			for (int y = -w / 2; y <= w / 2; y++) {
				int xx = x + w / 2;
				int yy = y + w / 2;
				double val = (1 / (2 * CV_PI * sigma * sigma)) * exp(-((pow(x, 2) + pow(y, 2)) / (2 * sigma*sigma)));
				kernel(yy, xx) = val;
				sum += val;
			}
		}
		convolute(src, srcF, w, kernel, 4, sum);

		//Gradient magnitude and orientation
		Mat_<double> Gx(src.rows, src.cols, (double)0.0f);
		Mat_<double> kernelGx(3, 3, 0.0f);
		kernelGx(0, 0) = -1.0f; kernelGx(0, 2) = 1.0f;
		kernelGx(1, 0) = -2.0f; kernelGx(1, 2) = 2.0f;
		kernelGx(2, 0) = -1.0f; kernelGx(2, 2) = 1.0f;
		convoluteDouble(srcF, Gx, 3, kernelGx);

		Mat_<double> Gy(src.rows, src.cols, 0.0f);
		Mat_<double> kernelGy(3, 3, (double)0.0f);
		kernelGy(0, 0) = 1; kernelGy(2, 0) = -1;
		kernelGy(0, 1) = 2; kernelGy(2, 1) = -2;
		kernelGy(0, 2) = 1; kernelGy(2, 2) = -1;
		convoluteDouble(srcF, Gy, 3, kernelGy);

		Mat_<double> _G(src.rows, src.cols, 0.0f);
		Mat_<double> phi(src.rows, src.cols, 0.0f);
		Mat_<uchar> G(src.rows, src.cols, 0.0f);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				_G(i, j) = sqrt((Gx(i, j) * Gx(i, j)) + (Gy(i, j) * Gy(i, j)))/(4 * sqrt(2));
				phi(i, j) = atan2(Gy(i, j), Gx(i, j));
				G(i, j) = _G(i, j) / (4 * sqrt(2));
			}
		}

		Mat_<uchar> Gs(src.rows, src.cols, 0.0f);
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				double angle = phi(i, j);
				double val = _G(i, j);
				if (angle >= -CV_PI / 8 && angle < CV_PI / 8 || angle >= 7*CV_PI / 8 || angle < -7*CV_PI/8) {
					if (_G(i, j - 1) <= val && _G(i, j + 1) <= val)
						Gs(i, j) = val;
					else Gs(i, j) = 0;
				}
				else if (angle >= CV_PI / 8 && angle < 3 * CV_PI / 8 || angle >= -7 * CV_PI / 8 && angle < -5 * CV_PI / 8) {
					if (_G(i + 1, j - 1) <= val && _G(i - 1, j + 1) <= val)
						Gs(i, j) = val;
					else Gs(i, j) = 0;
				}
				else if (angle >= 3 * CV_PI / 8 && angle < 5 * CV_PI / 8 || angle >= -5 * PI / 8 && angle < -3 * CV_PI / 8) {
					if (_G(i - 1, j) <= val && _G(i + 1, j) <= val)
						Gs(i, j) = val;
					else Gs(i, j) = 0;
				}
				else {
					if (_G(i - 1, j - 1) <= val && _G(i + 1, j + 1) <= val)
						Gs(i, j) = val;
					else Gs(i, j) = 0;
				}
			}
		}

		imshow("srcF", srcF);
		imshow("G", G);
		imshow("Gs", Gs);

		int histo[255] = { 0 };
		calcHisto(Gs, histo);
		float p = 0.1; float k = 0.4;
		int noEdgePixels = p * (Gs.rows * Gs.cols - histo[0]);
		int tVal = 0;
		int tHigh;
		for (tHigh = 254; tHigh >= 0 && tVal < noEdgePixels; tHigh--) {
			tVal += histo[tHigh];
		}
		int tLow = k * tHigh;

		printf("%d %d\n", tLow, tHigh);
		for (int i = 0; i < Gs.rows; i++) {
			for (int j = 0; j < Gs.cols; j++) {
				if (Gs(i, j) <= tLow) {
					Gs(i, j) = 0;
				}
				else if (Gs(i, j) <= tHigh) {
					Gs(i, j) = 127;
				}
				else Gs(i, j) = 255;
			}
		}
		imshow("Thresholded", Gs);
		
		int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
		for (int i = 0; i < Gs.rows; i++) {
			for (int j = 0; j < Gs.cols; j++) {
				if (Gs(i, j) == 255) {
					std::queue <Point> q;
					q.push(Point(i, j));
					while (!q.empty()) {
						Point p = q.front();
						q.pop();

						for (int k = 0; k < 8; k++) {
							int x = p.y + dx[k];
							int y = p.x + dy[k];
							if (pointInside(y, x, Gs.rows, Gs.cols) && Gs(y, x) == 127) {
								Gs(y, x) = 255;
								Point neighbor(y, x);
								q.push(neighbor);
							}
						}

					}
				}
			}
		}
		for (int i = 0; i < Gs.rows; i++) {
			for (int j = 0; j < Gs.cols; j++) {
				if (Gs(i, j) == 127) {
					Gs(i, j) = 0;
				}
			}
		}
		imshow("Final", Gs);
	}
}
int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf("10 - Change grey levels\n");
		printf("11 - Four squares\n");
		printf("12 - Inverse matrix\n");
		printf("13 - Split channels\n");
		printf("14 - Color to grayscale\n");
		printf("15 - Grayscale to black and white\n");
		printf("16 - RGB to HSV\n");
		printf("17 - Intensity histogram\n");
		printf("18 - Multilevel thresholding\n");
		printf("19 - Floyd-Steinberg dithering\n");
		printf("20 - Geometrical features of an object\n");
		printf("21 - Label components - Breadth First\n");
		printf("22 - Label components - Two Pass\n");
		printf("23 - Border tracing\n");
		printf("24 - Contour reconstruction\n");
		printf("25 - Image zooming\n");
		printf("26 - Dilation\n");
		printf("27 - Erode\n");
		printf("28 - Opening\n");
		printf("29 - Closing\n");
		printf("30 - Boundary extraction\n");
		printf("31 - Area filling\n");
		printf("32 - Histogram\n");
		printf("33 - Spatial domain\n");
		printf("34 - Frequency domain\n");
		printf("35 - Median filter\n");
		printf("36 - 2D Gaussian Filter\n");
		printf("37 - 2x1D Gaussian Filter\n");
		printf("38 - Canny edge detection\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		float *normalHisto;
		Mat_ <uchar> pic;
		int maxList[256] = { 0 };
		Mat_ <uchar> newPic;
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				changeGreyLevels();
				break;
			case 11:
				fourSquares();
				break;
			case 12:
				inverseMatrix();
				break;
			case 13:
				splitRGB();
				break;
			case 14:
				colorToGrayscale();
				break;
			case 15:
				grayscaleToBW(127);
				break;
			case 16:
				rgbToHsv();
				break;
			case 17:
				normalHisto = (float*)malloc(sizeof(float) * 256);
				intensityHistogram(normalHisto, true);
				break;
			case 18:
				normalHisto = (float*)malloc(sizeof(float) * 256);
				pic = intensityHistogram(normalHisto, false);
				multilevelThresholding(normalHisto, pic, maxList, true);
				break;
			case 19:
				normalHisto = (float*)malloc(sizeof(float) * 256);
				pic = intensityHistogram(normalHisto, false);
				multilevelThresholding(normalHisto, pic, maxList, false);
				floydSteinbergDithering(pic, maxList);
				break;
			case 20:
				geometrical_features();
				break;
			case 21:
				labelBF();
				break;
			case 22:
				labelTwoPass();
				break;
			case 23:
				borderTracing();
				break;
			case 24:
				contourReconstruction();
				break;
			case 25:
				zoomImage();
				break;
			case 26:
				printf("N=");
				int n;
				scanf("%d", &n);
				//dilate(n);
				break;
			case 27:
				printf("N=");
				scanf("%d", &n);
				//erode(n);
				break;
			case 28:
				opening();
				break;
			case 29:
				closing();
				break;
			case 30:
				extractBoundary();
				break;
			case 31:
				fillArea();
				break;
			case 32:
				statisticalProperties();
				break;
			case 33:
				meanFilters();
				break;
			case 34:
				fourier();
				break;
			case 35:
				int k;
				printf("K=");
				scanf("%d", &k);
				medianFilter(2*k + 1);
				break;
			case 36:
				float sigma;
				printf("Sigma=");
				scanf("%f", &sigma);
				gaussian2DFilter(sigma);
				break;
			case 37:
				printf("Sigma=");
				scanf("%f", &sigma);
				gaussian2x1DFilter(sigma);
				break;
			case 38:
				canny();
				break;
		}
	}
	while (op!=0);
	return 0;
}