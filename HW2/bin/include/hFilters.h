#pragma once
#include <vector>
#include <opencv2\opencv.hpp>
#include <algorithm>
#include "cvCtrl.h"

namespace fil {
	namespace kernel { enum Type { box, bilateral, guided }; };
	namespace Use { enum Type { normal, test }; };
	namespace skip { enum Type { yes, no }; };
	namespace destroy { enum Type { yes, no }; };
	namespace display { enum Type { normal, test }; };
	float PSNR(cv::Mat clean, cv::Mat filtered);
}

class hFilters {
public:

private:
	//grey matType=0; color matType=1
	int matType;
	int dataIndex;
	cv::Mat *data; int numData;
	std::vector<const char*> windowList;
	cv::Mat *testData;
public:
	hFilters(); ~hFilters();
	hFilters(char **fileNames, int numData, int MatType=0);



public:
	cv::Mat GuidedFilterG(int input, int guidance,
		int kernalSizeX, int kernalSizeY, float sigmaX, float sigmaY, float e);
	cv::Mat hFilters::GuidedFilterG(cv::Mat input, cv::Mat guidance,
		int kernelSizeX, int kernelSizeY, float sigmaX, float sigmaY, float e);

//	cv::Mat GuidedFilterG(cv::Mat p, cv::Mat I, 
//		int kernalSizeX, int kernalSizeY, float sigmaX, float sigmaY, float e);

	void loadImageRGBtoData(char **fileNames, int numData, int indBegin, int indEnd);

	cv::Mat weightedMedian(int input, int kType, int dataUsed, int kernelSizeX = 5, int kernelSizeY = 5,
		float sigmaX = 3, float sigmaY = 3, int guidance = -1,float e=0.5);

public:
	//displayImage: flag=0 destroy window after show, flag=1 don't destroy
	void displayImage(const char *windowName, int index = 0, int skip = 0, int flag=0, int display=0);
	void destroyWindow(const char*windowName);
	


public:
	inline void nextIndex() {
		this->dataIndex++;
		if (this->dataIndex == this->numData)
			this->dataIndex = 0;
	}
	inline void prevIndex() {
		this->dataIndex--;
		if (this->dataIndex == -1)
			this->dataIndex = this->numData-1;
	}

	inline const int getIndex() {
		return this->dataIndex;
	}
	inline const cv::Mat getImage(int index)
	{
		return this->data[index];
	}
	inline const cv::Mat getTestImage(int index)
	{
		return this->testData[index];
	}
};