#include <Windows.h>
#include <time.h>

#include <iostream>
#include <math.h>
#include <GL/glut.h>

#include <opencv2\opencv.hpp>

#include "FreeImage.h" //for reading Data

#include "Camera.h"

#include "glovar.h"
#include "hFilters.h"
#include "cvCtrl.h"


#define sysPause system("pause>nul")

using namespace cv;


int PrintRuntime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER f) {
	__int64 ms_interval = (end.QuadPart - start.QuadPart) / (f.QuadPart / 1000);
	__int64 micro_interval = (end.QuadPart - start.QuadPart) / (f.QuadPart / 1000000);
	printf("Millisecond: %d, microsecond: %d\n", (int)ms_interval, (int)micro_interval);

	return (int)ms_interval;
	//	printf("Millisecond: %d\n", (int)ms_interval);

}



int main(int argc, char** argv)
{
	LARGE_INTEGER start, end, f;
	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&start);
	QueryPerformanceCounter(&end);
	int runtime = PrintRuntime(start, end, f);
	float psigmaX, psigmaY, pkernelX, pkernelY, e;

	psigmaX = 3.f;
	psigmaY = 3.f;
	pkernelX = 5.f;
	pkernelY = 5.f;
	e = 0.5f;

	std::cout << "Input kernelsize x y (default 5, 5, must be odd)\n";
	std::cin >> pkernelX >> pkernelY;
	if ((pkernelX < 1) || (((int)pkernelX % 2) == 0))
	{
		pkernelX = 5.f;
		std::cout << "wront kernelX value setting default\n";
	}
	if ((pkernelY < 1) || (((int)pkernelY % 2) == 0))
	{
		pkernelY = 5.f;
		std::cout << "wront kernelY value setting default\n";
	}

	std::vector<float> rgbPSNRvalues;

	std::cout << "opencv version: " << CV_VERSION << std::endl;
	//Image Loading
	hFilters gFilterGrey(fileName, 6);
	hFilters gFilterColor(fileName, 6, 1);
	hFilters gFilterRGB;

	gFilterRGB.loadImageRGBtoData(fileName, 3, 3, 5);
	gFilterGrey.displayImage("Original_girl", 0);
	gFilterColor.displayImage("Original_Monkey_Color", 4, 0);
	gFilterRGB.displayImage("RGB-onechannel", 0, fil::skip::yes, fil::destroy::yes);
	std::cout << "proceeding to filtering" << std::endl;
#pragma region guided
	cv::Mat gGn1, gGn2, gMn1, gMn2, cGn1, cGn2, cMn1, cMn2;
	QueryPerformanceCounter(&start);
	gGn1 = gFilterGrey.GuidedFilterG(1, 1, pkernelX, pkernelY, psigmaX, psigmaY, e);
	gGn2 = gFilterGrey.GuidedFilterG(2, 2, pkernelX, pkernelY, psigmaX, psigmaY, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);
	QueryPerformanceCounter(&start);
	gMn1 = gFilterGrey.GuidedFilterG(4, 4, pkernelX, pkernelY, psigmaX, psigmaY, e);
	gMn2 = gFilterGrey.GuidedFilterG(5, 5, pkernelX, pkernelY, psigmaX, psigmaY, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);
	QueryPerformanceCounter(&start);
	cGn1 = gFilterColor.GuidedFilterG(1, 1, pkernelX, pkernelY, psigmaX, psigmaY, e);
	cGn2 = gFilterColor.GuidedFilterG(2, 2, pkernelX, pkernelY, psigmaX, psigmaY, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);
	QueryPerformanceCounter(&start);
	cMn1 = gFilterColor.GuidedFilterG(4, 4, pkernelX, pkernelY, psigmaX, psigmaY, e);
	cMn2 = gFilterColor.GuidedFilterG(5, 5, pkernelX, pkernelY, psigmaX, psigmaY, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);
	gFilterGrey.displayImage("gGo", 0, fil::skip::yes, fil::destroy::no);
	gFilterGrey.displayImage("gMo", 3, fil::skip::yes, fil::destroy::no);
	gFilterColor.displayImage("cMo", 3, fil::skip::yes, fil::destroy::no);
	displayImage("gGn1", gGn1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("gGn2", gGn2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("gMn1", gMn1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("gMn2", gMn2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("cGn1", cGn1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("cGn2", cGn2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("cMn1", cMn1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("cMn2", cMn2, cvCtrl::skip::yes, cvCtrl::destroy::no);

	cv::Mat gGo = gFilterGrey.getImage(0);
	cv::Mat gMo = gFilterGrey.getImage(3);
	rgbPSNRvalues.push_back(fil::PSNR(gGo, gGn1));
	rgbPSNRvalues.push_back(fil::PSNR(gGo, gGn2));
	rgbPSNRvalues.push_back(fil::PSNR(gMo, gMn1));
	rgbPSNRvalues.push_back(fil::PSNR(gMo, gMn2));
	
	std::cout << "Press ESC to continue to one-channel RGB\n";
	cv::waitKey();

	rgbPSNRvalues.clear();
	cv::destroyAllWindows();

#pragma endregion


#pragma region guidedRGB
	int rgbRuntime[6];

	std::vector<cv::Mat> n1;
	std::vector<cv::Mat> n2;

	QueryPerformanceCounter(&start);
	n1.push_back(gFilterRGB.GuidedFilterG(3, 0, pkernelX, pkernelY, psigmaX, psigmaY, e));
	QueryPerformanceCounter(&end);
	rgbRuntime[0] = PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	n1.push_back(gFilterRGB.GuidedFilterG(4, 1, pkernelX, pkernelY, psigmaX, psigmaY, e));
	QueryPerformanceCounter(&end);
	rgbRuntime[1] = PrintRuntime(start, end, f);
	
	QueryPerformanceCounter(&start);
	n1.push_back(gFilterRGB.GuidedFilterG(5, 2, pkernelX, pkernelY, psigmaX, psigmaY, e));
	QueryPerformanceCounter(&end);
	rgbRuntime[2] = PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	n2.push_back(gFilterRGB.GuidedFilterG(6, 0, pkernelX, pkernelY, psigmaX, psigmaY, e));
	QueryPerformanceCounter(&end);
	rgbRuntime[3] = PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	n2.push_back(gFilterRGB.GuidedFilterG(7, 1, pkernelX, pkernelY, psigmaX, psigmaY, e));
	QueryPerformanceCounter(&end);
	rgbRuntime[4] = PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	n2.push_back(gFilterRGB.GuidedFilterG(8, 2, pkernelX, pkernelY, psigmaX, psigmaY, e));
	QueryPerformanceCounter(&end);
	rgbRuntime[5] = PrintRuntime(start, end, f);

	gFilterColor.displayImage("originalMonkeyN1", 4, fil::skip::yes, fil::destroy::no);
	gFilterColor.displayImage("originalMonkeyN2", 5, fil::skip::yes, fil::destroy::no);

	displayImage("n1R", n1[0], cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("n1G", n1[1], cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("n1B", n1[2], cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("n2R", n2[0], cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("n2G", n2[1], cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("n2B", n2[2], cvCtrl::skip::yes, cvCtrl::destroy::no);
	cv::Mat n1RGB;					cv::Mat n2RGB;
	cv::merge(n1, n1RGB);			cv::merge(n2, n2RGB);

	displayImage("n1RGB", n1RGB, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("n2RGB", n2RGB, cvCtrl::skip::yes, cvCtrl::destroy::no);


	cv::Mat cColor;
	cv::Mat n1Color = gFilterColor.getImage(4);
	cv::Mat n2Color = gFilterColor.getImage(5);
	std::vector<cv::Mat> cRGB, n1RGBo, n2RGBo;
	cRGB.push_back(gFilterRGB.getImage(0));cRGB.push_back(gFilterRGB.getImage(1));cRGB.push_back(gFilterRGB.getImage(2));
	n1RGBo.push_back(gFilterRGB.getImage(3)); n1RGBo.push_back(gFilterRGB.getImage(4)); n1RGBo.push_back(gFilterRGB.getImage(5));
	n2RGBo.push_back(gFilterRGB.getImage(6)); n2RGBo.push_back(gFilterRGB.getImage(7)); n2RGBo.push_back(gFilterRGB.getImage(8));

	cv::merge(cRGB, cColor);
	rgbPSNRvalues.push_back(fil::PSNR(cRGB[0], n1RGBo[0])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[1], n1RGBo[1])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[2], n1RGBo[2]));
	rgbPSNRvalues.push_back(fil::PSNR(cRGB[0], n2RGBo[0])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[1], n2RGBo[1])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[2], n2RGBo[2]));
	rgbPSNRvalues.push_back(fil::PSNR(cRGB[0], n1[0])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[1], n1[1])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[2], n1[2]));
	rgbPSNRvalues.push_back(fil::PSNR(cRGB[0], n2[0])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[1], n2[1])); rgbPSNRvalues.push_back(fil::PSNR(cRGB[2], n2[2]));
	std::cout << "Press Anykey to continue to weighted Median\n";
	cv::waitKey();
	rgbPSNRvalues.clear();
	cv::destroyAllWindows();

#pragma endregion


#pragma region weighted median
	cv::Mat wmBox1, wmBox2, wmBi1, wmBi2, wmG1, wmG2;
	gFilterGrey.displayImage("testData", 0, fil::skip::yes, fil::destroy::yes, fil::display::test);
	pkernelX, pkernelY, psigmaX, psigmaY, e;
	QueryPerformanceCounter(&start);
	wmBox1 = gFilterGrey.weightedMedian(4, fil::kernel::box, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	wmBox2 = gFilterGrey.weightedMedian(5, fil::kernel::box, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	wmBi1 = gFilterGrey.weightedMedian(4, fil::kernel::bilateral, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	wmBi2 = gFilterGrey.weightedMedian(5, fil::kernel::bilateral, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	wmG1 = gFilterGrey.weightedMedian(4, fil::kernel::guided, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY, 4, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	wmG2 = gFilterGrey.weightedMedian(5, fil::kernel::guided, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY, 5, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);
	wmBox1.convertTo(wmBox1, CV_32FC1, 1.f / 255.f);
	wmBox2.convertTo(wmBox2, CV_32FC1, 1.f / 255.f);
	wmBi1.convertTo(wmBi1, CV_32FC1, 1.f / 255.f);
	wmBi2.convertTo(wmBi2, CV_32FC1, 1.f / 255.f);
	wmG1.convertTo(wmG1, CV_32FC1, 1.f / 255.f);
	wmG2.convertTo(wmG2, CV_32FC1, 1.f / 255.f);
	displayImage("wmBox1", wmBox1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("wmBox2", wmBox2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("wmBi1", wmBi1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("wmBi2", wmBi2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("wmG1", wmG1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("wmG2", wmG2, cvCtrl::skip::yes, cvCtrl::destroy::no);

	cv::Mat cleanImage = gFilterGrey.getImage(3);
	rgbPSNRvalues.push_back(fil::PSNR(cleanImage, wmBox1));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImage, wmBox2));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImage, wmBi1));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImage, wmBi2));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImage, wmG1));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImage, wmG2));

	std::cout << "press ESC to Girl pic\n";
	cv::waitKey();
	cv::destroyAllWindows();

#pragma endregion

#pragma region weighted medain Girl
	cv::Mat GwmBox1, GwmBox2, GwmBi1, GwmBi2, GwmG1, GwmG2;
	gFilterGrey.displayImage("testData", 0, fil::skip::yes, fil::destroy::yes, fil::display::test);
	pkernelX, pkernelY, psigmaX, psigmaY, e;
	QueryPerformanceCounter(&start);
	GwmBox1 = gFilterGrey.weightedMedian(1, fil::kernel::box, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	GwmBox2 = gFilterGrey.weightedMedian(2, fil::kernel::box, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	GwmBi1 = gFilterGrey.weightedMedian(1, fil::kernel::bilateral, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	GwmBi2 = gFilterGrey.weightedMedian(2, fil::kernel::bilateral, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	GwmG1 = gFilterGrey.weightedMedian(1, fil::kernel::guided, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY, 1, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);

	QueryPerformanceCounter(&start);
	GwmG2 = gFilterGrey.weightedMedian(2, fil::kernel::guided, fil::Use::normal, pkernelX, pkernelY, psigmaX, psigmaY, 2, e);
	QueryPerformanceCounter(&end);
	PrintRuntime(start, end, f);
	GwmBox1.convertTo(GwmBox1, CV_32FC1, 1.f / 255.f);
	GwmBox2.convertTo(GwmBox2, CV_32FC1, 1.f / 255.f);
	GwmBi1.convertTo(GwmBi1, CV_32FC1, 1.f / 255.f);
	GwmBi2.convertTo(GwmBi2, CV_32FC1, 1.f / 255.f);
	GwmG1.convertTo(GwmG1, CV_32FC1, 1.f / 255.f);
	GwmG2.convertTo(GwmG2, CV_32FC1, 1.f / 255.f);
	displayImage("GwmBox1", GwmBox1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("GwmBox2", GwmBox2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("GwmBi1", GwmBi1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("GwmBi2", GwmBi2, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("GwmG1", GwmG1, cvCtrl::skip::yes, cvCtrl::destroy::no);
	displayImage("GwmG2", GwmG2, cvCtrl::skip::yes, cvCtrl::destroy::no);

	cv::Mat cleanImageg = gFilterGrey.getImage(0);
	rgbPSNRvalues.push_back(fil::PSNR(cleanImageg, GwmBox1));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImageg, GwmBox2));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImageg, GwmBi1));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImageg, GwmBi2));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImageg, GwmG1));
	rgbPSNRvalues.push_back(fil::PSNR(cleanImageg, GwmG2));

	std::cout << "press ESC to end\n";
	cv::waitKey();
	cv::destroyAllWindows();

#pragma endregion


	cv::destroyAllWindows();
	return 0;
}