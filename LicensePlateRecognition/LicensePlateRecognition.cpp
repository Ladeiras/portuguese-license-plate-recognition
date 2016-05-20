/**
	FEUP 2014
	[EIC0104] Computer Vision

	License Plate Recognition
	LicensePlateRecognition.cpp

	@author Jivitha Anand
	@author João Ladeiras
*/

#include "stdafx.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

//Globals
#define IMG_PATH "../img/"
#define TEMPL_PATH "../templates/"

#define NR_OF_CHARS 37

Scalar yellow_start = Scalar(20, 100, 100);
Scalar yellow_end = Scalar(30, 255, 255);

Scalar blue_start = Scalar(90, 50, 50);
Scalar blue_end = Scalar(130, 255, 255);

//Functions
Mat findPlate(string);
Mat findContour(Mat);
int findLargestAreaContourIndex(Mat, vector<vector<Point> >);
int findLargestArcContourIndex(Mat, vector<vector<Point> >);
vector<Vec4i> createHoughLines(Mat);
Point2f computeIntersect(Vec4i, Vec4i);
vector<Vec4i> findQuadLines(Mat, vector<Vec4i>);
void combineLines(int, int, vector<Vec4i> &, vector<Vec4i> &, 
	vector<vector<Vec4i> > &);
vector<Vec4i> findBestQuadLine(Mat, vector<vector<Vec4i> >);
float euclideanDistance(Point2f&, Point2f&);
void selectColorRectangles(Mat, vector<vector<Point> >&, 
	vector<vector<Point> >&);
Mat transformQuadriteral(Mat, vector<Vec4i> lines);
void sortCorners(vector<Point2f>&, Point2f);
void findCharacters(Mat);
string templateMatching(vector<Mat>,double&);
string getTmplFileName(int);
string nrToChar(int nr);

int main()
{
	/// Find Plate
	//Mat image = findPlate("img_01.jpg");
	Mat image = findPlate("img_02.jpg");
	//Mat image = findPlate("img_03.jpg");
	//Mat image = findPlate("img_04.jpg");
	imshow("Image", image);

	/// Find Contour
	Mat contour = findContour(image);
	imshow("Contour", contour);

	/// Hough Lines
	vector<Vec4i> lines = createHoughLines(contour);

	/// Persperctive Transform
	Mat plate = transformQuadriteral(image, lines);
	imshow("Plate", plate);

	/// Find Characters
	findCharacters(plate);

	waitKey(0);
	return 0;
}

/**
	Finds the location of one plate in an image and returns a cropped image
	with the area of interest.

	@param imageName The name of the image file
	@return plate The matrix of the result image
*/
Mat findPlate(string imageName)
{
	Mat cvtImage, eImgY, dImgY, eImgB, dImgB;
	Mat tmpImgY, tmpImgB;

	/// Read Images From File
	Mat src = imread(IMG_PATH + imageName);
	Mat image = imread(IMG_PATH + imageName);
	cvtColor(image, cvtImage, CV_BGR2HSV, 0);

	tmpImgY = cvtImage;
	tmpImgB = cvtImage;

	/// Get Image Thresh
	Mat thrImgY(tmpImgY.size(), tmpImgY.type());
	Mat thrImgB(tmpImgB.size(), tmpImgB.type());

	/// Yellow Color
	inRange(tmpImgY, yellow_start, yellow_end, thrImgY);
	/// Blue Color
	inRange(tmpImgB, blue_start, blue_end, thrImgB);

	/// Elements For The Morfological Transformation
	Mat elementY(5, 5, CV_8U, Scalar(1));
	Mat elementB(5, 5, CV_8U, Scalar(1));

	/// Contours
	vector<vector<Point> > contoursY;
	vector<vector<Point> > contoursB;

	imshow("Before Morphological Transform", thrImgY);

	/// Handle Yellow Zones
	int nErodeY = 1;
	int nDilateY = 6;
	while (contoursY.size() == 0)
	{
		if (nErodeY < 0)
			break;
		/// Eroding Yellow Zones
		erode(thrImgY, eImgY, elementY);
		for (int h = 0; h<nErodeY; h++)
		{
			erode(eImgY, eImgY, elementY);
		}
		//imshow("Dilate", eImgY);

		/// Dilating Yellow Zones
		dilate(eImgY, dImgY, elementY);
		for (int h = 0; h<nDilateY; h++)
		{
			dilate(dImgY, dImgY, elementY);
		}
		imshow("After Morphological Transform", dImgY);

		/// Find Contours In Resulting Image
		findContours(dImgY, contoursY, CV_RETR_EXTERNAL, 
			CV_CHAIN_APPROX_SIMPLE);

		/// If Nothing Found Changing Eroding Values
		if (contoursY.size() == 0)
		{
			nErodeY--;
			inRange(tmpImgY, yellow_start, yellow_end, thrImgY);
		}
	}
	cout << "\nnErodeY = " << nErodeY << endl;
	cout << "nDilateY = " << nDilateY << endl;
	cout << "\n";

	vector<vector<Point> > contours_polyY(contoursY.size());
	vector<Rect> boundRectY(contoursY.size());

	/// Approximate Polygons
	for (unsigned int i = 0; i < contoursY.size(); i++)
	{
		approxPolyDP(Mat(contoursY[i]), contours_polyY[i], 25, true);
		boundRectY[i] = boundingRect(Mat(contours_polyY[i]));
	}

	/// Rectangle With The Found Areas
	for (unsigned int j = 0; j < contoursY.size(); j++)
	{
		rectangle(image, boundRectY[j], Scalar(0, 255, 255), 2);
	}

	/// Handle Blue Zones
	int nErodeB = 1;
	int nDilateB = 6;
	while (contoursB.size() == 0)
	{
		if (nErodeB < 0)
			break;
		/// Eroding Blue Zones
		erode(thrImgB, eImgB, elementB);
		for (int h = 0; h<nErodeB; h++)
		{
			erode(eImgB, eImgB, elementB);
		}
		//imshow("Dilate", eImgB);

		/// Dilating Blue Zones
		dilate(eImgB, dImgB, elementB);
		for (int h = 0; h<nDilateB; h++)
		{
			dilate(dImgB, dImgB, elementB);
		}
		//imshow("Blue", dImgB);

		/// Find Contours In Resulting Image
		findContours(dImgB, contoursB, CV_RETR_EXTERNAL, 
			CV_CHAIN_APPROX_SIMPLE);

		/// If Nothing Found Changing Eroding Values
		if (contoursB.size() == 0)
		{
			nErodeB--;
			inRange(tmpImgB, blue_start, blue_end, thrImgB);
		}
	}
	cout << "\nnErodeB = " << nErodeB << endl;
	cout << "nDilateB = " << nDilateB << endl;
	cout << "\n";

	vector<vector<Point> > contours_polyB(contoursB.size());
	vector<Rect> boundRectB(contoursB.size());

	/// Approximate Polygons
	for (unsigned int i = 0; i < contoursB.size(); i++)
	{
		approxPolyDP(Mat(contoursB[i]), contours_polyB[i], 25, true);
		boundRectB[i] = boundingRect(Mat(contours_polyB[i]));
	}

	/// Rectangle With The Found Areas
	for (unsigned int j = 0; j < contoursB.size(); j++)
	{
		rectangle(image, boundRectB[j], Scalar(255, 0, 0), 2);
	}

	/// Show Result
	imshow("Color Detect", image);

	/// Erase Smaller Areas
	cout << "SizeY: " << contours_polyY.size() << endl;
	for (unsigned int i = 0; i < contours_polyY.size(); i++)
	{
		if (contours_polyY[i].size() < 3)
			contours_polyY.erase(contours_polyY.begin() + i, 
			contours_polyY.begin() + i);
		else
		{
			for (unsigned int j = 0; j < contours_polyY[i].size(); j++)
				cout << contours_polyY[i][j] << endl;
			cout << "\n";
		}
	}

	cout << "SizeB: " << contours_polyB.size() << endl;
	for (unsigned int i = 0; i < contours_polyB.size(); i++)
	{
		if (contours_polyB[i].size() < 3)
			contours_polyB.erase(contours_polyB.begin() + i, 
			contours_polyB.begin() + i);
		else
		{
			for (unsigned int j = 0; j < contours_polyB[i].size(); j++)
				cout << contours_polyB[i][j] << endl;
			cout << "\n";
		}
	}

	/// Filtering The Rectangles From The Plate
	selectColorRectangles(image, contours_polyY, contours_polyB);

	Mat plate;
	int x1 = image.cols;
	int x2 = 0;
	int y1 = image.rows;
	int y2 = 0;

	/// Finding The Points Max And Min Values
	for (unsigned int i = 0; i < contours_polyY[0].size(); i++)
	{
		if (contours_polyY[0][i].x < x1)
			x1 = contours_polyY[0][i].x;
		if (contours_polyY[0][i].y < y1)
			y1 = contours_polyY[0][i].y;

		if (contours_polyY[0][i].x > x2)
			x2 = contours_polyY[0][i].x;
		if (contours_polyY[0][i].y > y2)
			y2 = contours_polyY[0][i].y;
	}

	for (unsigned int i = 0; i < contours_polyB[0].size(); i++)
	{
		if (contours_polyB[0][i].x < x1)
			x1 = contours_polyB[0][i].x;
		if (contours_polyB[0][i].y < y1)
			y1 = contours_polyB[0][i].y;

		if (contours_polyB[0][i].x > x2)
			x2 = contours_polyB[0][i].x;
		if (contours_polyB[0][i].y > y2)
			y2 = contours_polyB[0][i].y;
	}

	/// Expand The Area
	int expand = 10;
	x1 -= x1 - expand > 0 ? expand : 0;
	x2 += x2 + expand > image.cols ? 0 : expand;
	y1 -= y1 - expand > 0 ? expand : 0;
	y2 += y2 + expand > image.rows ? 0 : expand;

	cout << x1 << "," << x2 << "," << y1 << "," << y2 << endl;

	/// Crop The Image
	Rect rect = Rect(x1, y1, x2 - x1, y2 - y1);
	src(rect).copyTo(plate);

	return plate;
}

/**
	Selects the rectangles from the color area selection (blue and yellow) that
	best fits the properties of the position of both colors in a plate.

	@param image The matrix of the source image
	@param contours_polyY Information of all the yellow contours
	@param contours_polyB Information of all the blue contours
*/
void selectColorRectangles(Mat image, vector<vector<Point> > &contours_polyY, 
	vector<vector<Point> > &contours_polyB)
{
	cout << "Number of Yellows : " << contours_polyY.size() << endl;
	cout << "Number of Blues : " << contours_polyB.size() << endl;

	int counter = 0;
	int flag1 = false;
	int flag2 = false;

	int yIndex = 0;
	int bIndex = 0;

	vector<vector<Point> > tContours_polyY;
	vector<vector<Point> > tContours_polyB;

	/// All x's in Yellow are greater than x's in Blue & Some y's must be equal
	for (unsigned int i = 0; i < contours_polyY.size(); i++)
	{
		for (unsigned int j = 0; j < contours_polyB.size(); j++)
		{
			int yMin = image.rows;
			int yMax = 0;
			/// Calc max and min Y values
			for (unsigned int iy = 0; iy < contours_polyY[i].size(); iy++)
			{
				if (contours_polyY[i][iy].y < yMin)
					yMin = contours_polyY[i][iy].y;

				if (contours_polyY[i][iy].y > yMax)
					yMax = contours_polyY[i][iy].y;
			}

			/// Compare x's and y's
			for (unsigned int iy = 0; iy < contours_polyY[i].size(); iy++)
			{
				for (unsigned int ib = 0; ib < contours_polyB[j].size(); ib++)
				{
					
					if (contours_polyY[i][iy].x < contours_polyB[j][ib].x)
					{
						flag1 = true;
						break;
					}
					else if (contours_polyB[j][ib].y > yMin && contours_polyB[j][ib].y < yMax)
					{
						flag2 = true;
						//cout << "Possible combination : " << i << "," << j << endl;
					}
				}

				if (flag1)
					break;
			}

			if (!flag1 && flag2)
			{
				yIndex = i;
				bIndex = j;
				counter++;
				flag2 = false;
			}
			else 
				flag1 = false;
		}
	}

	tContours_polyY.push_back(contours_polyY[yIndex]);
	tContours_polyB.push_back(contours_polyB[bIndex]);

	contours_polyY = tContours_polyY;
	contours_polyB = tContours_polyB;

	cout << "Possible combinations: " << counter << endl;
}

/**
	Finds the plate contour that must be the contour with the largest area

	@param src Plate image matrix
	@return drawing Contour's grey scale image matrix
*/
Mat findContour(Mat src)
{
	int thresh = 50;
	Mat src_gray, canny_output;
	vector<vector<Point> > contours;
	int contourIndex;
	vector<Vec4i> hierarchy;

	/// Convert Image To Gray And Blur It
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// Detect Edges Using Canny
	Canny(src_gray, canny_output, thresh, thresh * 3, 3);

	/// Find Contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, 
		CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Apply Polygon Approximation
	vector<vector<Point> > contours_poly(contours.size());
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 25, true);
	}

	/// Find The Contour With The Largest Area
	contourIndex = findLargestAreaContourIndex(src, contours);
	
	/// Draw The Contour In A New Image
	Scalar contourColor;
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	
	Scalar color = Scalar(255, 255, 255);
	contourColor = color;
	drawContours(drawing, contours, contourIndex, color, 1, 8, hierarchy, 0, 
		Point());

	return drawing;
}

/**
	Finds the contour with the largest arc and returns its index in the vector

	@param src Source image matrix
	@param contours Vector containing all the contours
	@return largest_arc_index Index of the contour in the vector
*/
int findLargestArcContourIndex(Mat src, vector<vector<Point> > contours)
{
	double largest_arc = 0;
	int largest_arc_index;

	int maxArc = (src.cols + src.rows)*2;

	/// Find Largest contour
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		double a = arcLength(contours[i], true);

		if (a > largest_arc && a < maxArc)
		{
			largest_arc = a;
			largest_arc_index = i;
		}
	}

	return largest_arc_index;
}

/**
	Finds the contour with the largest area and returns its index in the vector

	@param src Source image matrix
	@param contours Vector containing all the contours
	@return largest_arc_index Index of the contour in the vector
*/
int findLargestAreaContourIndex(Mat src, vector<vector<Point> > contours)
{
	double largest_area = 0;
	int largest_area_index;

	/// Find Largest contour
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i], false);

		if (a > largest_area)
		{
			largest_area = a;
			largest_area_index = i;
		}
	}

	return largest_area_index;
}

/**
	Finds, extends and returns four Hough Lines in an image with a rectangular
	contour

	@param src Source image matrix
	@return lines Vector with the lines
*/
vector<Vec4i> createHoughLines(Mat src)
{
	vector<Vec4i> lines;
	Mat dst;

	/// Find all Hough Lines
	cvtColor(src, dst, CV_BGR2GRAY);
	HoughLinesP(dst, lines, 1, CV_PI / 180, 10, 30, 10);

	cout << "Number of Lines: " << lines.size() << endl;

	/// Extend Lines
	for (unsigned int i = 0; i < lines.size(); i++)
	{
		Vec4i v = lines[i];
		lines[i][0] = 0;
		lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2]) * -v[0] + v[1];
		lines[i][2] = src.cols;
		lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2]) * 
			(src.cols - v[2]) + v[3];
	}
	
	/// Find Four Lines Of The Rectangular Contour
	lines = findQuadLines(dst, lines);

	Mat drawing = src.clone();

	/// Draw lines
	for (unsigned int i = 0; i < lines.size(); i++)
	{
		Vec4i v = lines[i];
		line(drawing, Point(v[0], v[1]), Point(v[2], v[3]), CV_RGB(0, 255, 0));
	}

	imshow("Hough Lines", drawing);

	return lines;
}

/**
	Calculates the point in which two lines intersect

	@param a First vector
	@param b Second vector
	@return pt Calculated point
*/
Point2f computeIntersect(Vec4i a, Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], 
		x4 = b[2], y4 = b[3];

	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		Point2f pt;
		pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * 
			(x3 * y4 - y3 * x4)) / d;
		pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * 
			(x3 * y4 - y3 * x4)) / d;
		return pt;
	}
	else
		return Point2f(-1, -1);
}

/**
	Finds four lines of a given rectangular contour

	@param src Source image matrix
	@param lines Vector with the lines
	@return lines Vector with the four lines
*/
vector<Vec4i> findQuadLines(Mat src, vector<Vec4i> lines)
{
	vector<Vec4i> combination;
	vector<vector<Vec4i> > combinations;

	/// Combine All Lines In Sets Of Four
	combineLines(0, 4, lines, combination, combinations);

	cout << "Combinations " << lines.size() << " set of 4 : " << 
		combinations.size() << endl;

	/// Find The Best Set
	lines = findBestQuadLine(src, combinations);

	return lines;
}

/**
	Recursively combine a set of given lines in sets of four

	@param offset Offset
	@param lines Vector with the lines
	@param combination Vector wit a combination of lines
	@param combinations Vector with all combinations of lines
*/
void combineLines(int offset, int k, vector<Vec4i> &lines, 
	vector<Vec4i> &combination, vector<vector<Vec4i> > &combinations)
{
	//End
	if (k == 0)
	{
		vector<Vec4i> tCombination = combination;
		combinations.push_back(tCombination);
		return;
	}

	/// Combine
	for (unsigned int i = offset; i <= lines.size() - k; ++i)
	{
		combination.push_back(lines[i]);
		combineLines(i + 1, k - 1, lines, combination, combinations);
		combination.pop_back();
	}
}

/**
	Finds the combination of sets of four lines that best describe a 
	rectangular contour

	@param src Source image matrix
	@param combinations Vector with all combinations of four lines
	@return combination Vector of the best set of lines
*/
vector<Vec4i> findBestQuadLine(Mat src, vector<vector<Vec4i> > combinations)
{
	vector<Point2f> corners;
	double distance;
	double sumDistance = 0;
	double topDistance = 0;
	unsigned int topDistanceIndex = 0;

	for (unsigned int i = 0; i < combinations.size(); i++)
	{
		for (unsigned int j = 0; j < combinations[i].size(); j++)
		{
			for (unsigned int k = j + 1; k < combinations[i].size(); k++)
			{
				/// Calculate Corner Points Of The Set
				Point2f pt = computeIntersect(combinations[i][j], 
					combinations[i][k]);
				if (pt.x >= 0 && pt.y >= 0 && 
					pt.x <= src.cols && pt.y <= src.rows)
				{
					corners.push_back(pt);
				}
			}
		}

		//cout << "Number of corners : " << corners.size() << endl;

		/* If Four Corners Are Found Calculate Each Others Distance And Find
		   The Largest */
		if (corners.size() == 4)
		{
			for (unsigned int ic = 0; ic < corners.size(); ic++)
			{
				/*cout << "\nCoordinates of corner : " << corners[ic].x << "," 
				<< corners[ic].y << endl;*/
				for (unsigned int jc = ic + 1; jc < corners.size(); jc++)
				{
					distance = euclideanDistance(corners[ic], corners[jc]);
					sumDistance += distance;
					/*cout << "Distance between corner " << ic 
					<< " and corner " << jc << " is : " << distance << endl;*/
				}
			}

			//cout << "Total distance : " << sumDistance << endl;

			if (sumDistance > topDistance)
			{
				topDistance = sumDistance;
				topDistanceIndex = i;
			}

			sumDistance = 0;
		}

		corners.clear();
	}

	return combinations[topDistanceIndex];
}

/**
	Calculates the distance between two given points

	@param p First point
	@param q Second point
	@return r Distance between points
*/
float euclideanDistance(Point2f& p, Point2f& q)
{
	Point2f diff = p - q;
	return sqrt(diff.x*diff.x + diff.y*diff.y);
}

/**
	Transforms the perspective of an image given four corners of the 
	rectangular form

	@param src Source image matrix
	@param lines Set of four lines
	@return quad Warped image matrix
*/
Mat transformQuadriteral(Mat src, vector<Vec4i> lines)
{
	Point2f center(0, 0);

	/// Find Corners Between Lines
	vector<Point2f> corners;
	for (unsigned int i = 0; i < lines.size(); i++)
	{
		for (unsigned int j = i + 1; j < lines.size(); j++)
		{
			Point2f pt = computeIntersect(lines[i], lines[j]);
			if (pt.x >= 0 && pt.y >= 0 && pt.x <= src.cols && pt.y <= src.rows)
				corners.push_back(pt);
		}
	}

	cout << "Corners Size: " << corners.size() << endl;

	vector<Point2f> approx;
	approxPolyDP(Mat(corners), approx, arcLength(Mat(corners), true) * 0.02, 
		true);

	/// Calculate Image Center
	for (unsigned int i = 0; i < corners.size(); i++)
		center += corners[i];
	center *= (1. / corners.size());

	/// Sort Corners
	sortCorners(corners, center);

	Mat dst = src.clone();

	/// Draw Corner Points
	circle(dst, corners[0], 3, CV_RGB(255, 0, 0), 2);
	circle(dst, corners[1], 3, CV_RGB(0, 255, 0), 2);
	circle(dst, corners[2], 3, CV_RGB(0, 0, 255), 2);
	circle(dst, corners[3], 3, CV_RGB(255, 255, 255), 2);

	/// Draw Mass Center
	circle(dst, center, 3, CV_RGB(255, 255, 0), 2);

	imshow("Corners", dst);
	//waitKey(0);

	/// Create New Image With Plate Dimensions
	Mat quad = Mat::zeros(160, 720, CV_8UC3);

	vector<Point2f> quad_pts;
	quad_pts.push_back(Point2f(0, 0));
	quad_pts.push_back(Point2f(quad.cols, 0));
	quad_pts.push_back(Point2f(quad.cols, quad.rows));
	quad_pts.push_back(Point2f(0, quad.rows));

	/// Warp Perspective
	Mat transmtx = getPerspectiveTransform(corners, quad_pts);
	warpPerspective(src, quad, transmtx, quad.size());

	return quad;
}

/**
	Sorts corner points according to a center of a quadriteral

	@param corners Quadriteral corner points
	@param center Qadriteral central point
*/
void sortCorners(vector<Point2f>& corners, Point2f center)
{
	cout << "\nNumber of corners : " << corners.size() << endl;
	vector<Point2f> top, bot;

	for (unsigned int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}

	cout << "Center: " << center << endl;

	Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

/**
	Finds the characters in a license plate or partial license plate image

	@param src Source image matrix
*/
void findCharacters(Mat img)
{
	/// Character - 10.5% Of Image
	int charWidth = (int)img.cols*0.105;
	/// Separator - 8% Of Image
	int charBreak = (int)img.cols*0.08;

	int charWidth2 = (int)img.cols*0.11;
	int charBreak2 = (int)img.cols*0.07;
	int offset = (int)img.cols*0.04;

	vector<Mat> chars;
	string plate;
	double weight1, weight2;

	/// Crop Image In Parts
	Mat char1, char2, char3, char4, char5, char6;
	Rect rect1, rect2, rect3, rect4, rect5, rect6;
	
	/// Full Plate
	rect1 = Rect(charWidth, 0, charWidth, img.rows);
	img(rect1).copyTo(char1);
	chars.push_back(char1);

	rect2 = Rect(charWidth * 2, 0, charWidth, img.rows);
	img(rect2).copyTo(char2);
	chars.push_back(char2);

	rect3 = Rect(charWidth * 3 + charBreak, 0, charWidth, img.rows);
	img(rect3).copyTo(char3);
	chars.push_back(char3);

	rect4 = Rect(charWidth * 4 + charBreak, 0, charWidth, img.rows);
	img(rect4).copyTo(char4);
	chars.push_back(char4);

	rect5 = Rect(charWidth * 5 + charBreak * 2, 0, charWidth, img.rows);
	img(rect5).copyTo(char5);
	chars.push_back(char5);

	rect6 = Rect(charWidth * 6 + charBreak * 2, 0, charWidth, img.rows);
	img(rect6).copyTo(char6);
	chars.push_back(char6);

	/// Template Matching
	cout << "\nMethod 1" << endl;
	plate = templateMatching(chars, weight1);
	cout << "Total weight: " << weight1 << endl;
	chars.clear();

	/// Partial 1
	/*rect1 = Rect(offset, 0, charWidth2, img.rows);
	img(rect1).copyTo(char1);
	chars.push_back(char1);

	rect2 = Rect(offset + charWidth2, 0, charWidth2, img.rows);
	img(rect2).copyTo(char2);
	chars.push_back(char2);

	rect3 = Rect(offset + charWidth2 * 2 + charBreak2, 0, charWidth2, 
	img.rows);
	img(rect3).copyTo(char3);
	chars.push_back(char3);

	rect4 = Rect(offset + charWidth2 * 3 + charBreak2, 0, charWidth2, 
	img.rows);
	img(rect4).copyTo(char4);
	chars.push_back(char4);

	rect5 = Rect(offset + charWidth2 * 4 + charBreak2 * 2, 0, charWidth2, 
	img.rows);
	img(rect5).copyTo(char5);
	chars.push_back(char5);

	rect6 = Rect(offset + charWidth * 5 + charBreak2 * 2, 0, charWidth2, 
	img.rows);
	img(rect6).copyTo(char6);
	chars.push_back(char6);

	cout << "\nMethod 2" << endl;
	plate = templateMatching(chars, weight2);
	cout << "Total weight: " << weight2 << endl;
	chars.clear();*/

	/*imshow("Char1", char1);
	imshow("Char2", char2);
	imshow("Char3", char3);
	imshow("Char4", char4);
	imshow("Char5", char5);
	imshow("Char6", char6);*/

	/// Final License Plate
	plate = plate.substr(0, 2).append(" - ").append(plate.substr(2, 2)).
		append(" - ").append(plate.substr(4, 2));
	cout << "\nPlate : " << plate << endl;
	imshow("Plate", img);
}

/**
	Finds the characters in individual parts of a license plate using template
	matching

	@param chars Characters of license plate image matrices
	@param weight Sum of all the threshold used
	@return plate String of the final license plate
*/
string templateMatching(vector<Mat> chars, double &weight)
{
	string plate;
	bool flag = false;
	double dec = 0.05;
	weight = 0.0;

	for (unsigned int j = 0; j < chars.size(); j++)
	{
		/// Decreasing Threshold Until Getting A Result
		double thresholdValue = 1;
		while (thresholdValue > 0)
		{
			for (unsigned int i = 0; i < NR_OF_CHARS; i++)
			{
				/// Get Character Template
				string templName = getTmplFileName(i);
				Mat tpl = imread(TEMPL_PATH + templName);

				Mat gref, gtpl;
				cvtColor(chars[j], gref, CV_BGR2GRAY);
				cvtColor(tpl, gtpl, CV_BGR2GRAY);

				/// Match Template
				Mat res(chars[j].rows - tpl.rows + 1, 
					chars[j].cols - tpl.cols + 1, CV_32FC1);
				matchTemplate(gref, gtpl, res, CV_TM_CCOEFF_NORMED);
				threshold(res, res, thresholdValue, 1., CV_THRESH_TOZERO);

				double minval, maxval, threshold = thresholdValue;
				Point minloc, maxloc;
				minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);

				/// Threshold
				if (maxval >= threshold)
				{
					/*rectangle(chars[j], maxloc, Point(maxloc.x + tpl.cols, 
					maxloc.y + tpl.rows), CV_RGB(0, 255, 0), 2);*/
					//floodFill(res, maxloc, Scalar(0), 0, Scalar(.1), Scalar(1.));
					plate.append(nrToChar(i));
					weight += thresholdValue;
					cout << "Char " << j << ": " << nrToChar(i) << endl;
					flag = true;
					break;
				}
			}

			if (flag)
			{
				flag = false;
				break;
			}
			else thresholdValue -= dec;
		}
	}

	return plate;
}

/**
	Returns the name of the assigned character file

	@param nr Character index
	@return name File name
*/
string getTmplFileName(int nr)
{
	switch (nr)
	{
	case 0:
		return "0.jpg";
		break;
	case 1:
		return "1.jpg";
		break;
	case 2:
		return "2.jpg";
		break;
	case 3:
		return "3.jpg";
		break;
	case 4:
		return "4.jpg";
		break;
	case 5:
		return "5.jpg";
		break;
	case 6:
		return "6.jpg";
		break;
	case 7:
		return "7.jpg";
		break;
	case 8:
		return "8.jpg";
		break;
	case 9:
		return "9.jpg";
		break;
	case 10:
		return "A.jpg";
		break;
	case 11:
		return "B.jpg";
		break;
	case 12:
		return "C.jpg";
		break;
	case 13:
		return "D.jpg";
		break;
	case 14:
		return "E.jpg";
		break;
	case 15:
		return "F.jpg";
		break;
	case 16:
		return "G.jpg";
		break;
	case 17:
		return "H.jpg";
		break;
	case 18:
		return "I.jpg";
		break;
	case 19:
		return "J.jpg";
		break;
	case 20:
		return "K.jpg";
		break;
	case 21:
		return "L.jpg";
		break;
	case 22:
		return "M.jpg";
		break;
	case 23:
		return "N.jpg";
		break;
	case 24:
		return "O.jpg";
		break;
	case 25:
		return "P.jpg";
		break;
	case 26:
		return "Q.jpg";
		break;
	case 27:
		return "R.jpg";
		break;
	case 28:
		return "S.jpg";
		break;
	case 29:
		return "T.jpg";
		break;
	case 30:
		return "U.jpg";
		break;
	case 31:
		return "V.jpg";
		break;
	case 32:
		return "W.jpg";
		break;
	case 33:
		return "X.jpg";
		break;
	case 34:
		return "Y.jpg";
		break;
	case 35:
		return "Z.jpg";
		break;
	case 36:
		return "0.jpg";
		break;
	default:
		break;
	}

	return "0.jpg";
}

/**
	Returns the char of the assigned character index

	@param nr Character index
	@return char Character string
*/
string nrToChar(int nr)
{
	string filename = getTmplFileName(nr);
	return filename.substr(0, 1);
}

