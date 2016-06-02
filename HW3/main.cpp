#include <Windows.h>
#include <time.h>

#include <iostream>
#include <math.h>
#include <GL/glut.h>

#include <opencv2\opencv.hpp>

#include "FreeImage.h" //for reading Data

#include "Camera.h"

#include "glovar.h"
#include "stitchImage.h"
#include "cvCtrl.h"


#define sysPause system("pause>nul")



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

	stitchImage CstitchImage((const char**)fileName, 5);

	CstitchImage.extractDesc();
	CstitchImage.match();
	CstitchImage.ransac((int)4);
	CstitchImage.stitch();

	cv::destroyAllWindows();
	return 0;
}