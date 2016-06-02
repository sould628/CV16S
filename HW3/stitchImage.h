#pragma once
#include <vector>
#include <opencv2\opencv.hpp>
#include <algorithm>
#include "cvCtrl.h"

extern "C" {
#include <vl/generic.h>
#include <vl/sift.h>
#include <vl/kdtree.h>
}


#define ST_PI 3.14159265358979323846
#define pause system("pause")

struct matchPair {
	VlSiftKeypoint *key1;
	VlSiftKeypoint *key2;



	matchPair() {};
	matchPair(VlSiftKeypoint *k1,VlSiftKeypoint *k2) {
		key1 = k1;
		key2 = k2;
	}
};

struct desc{
	float descriptor[128];
	VlSiftKeypoint *key;

	desc(float desc[128], VlSiftKeypoint *key)
	{
		for (int i = 0; i < 128; i++)
			descriptor[i] = desc[i];
		this->key = key;
	}
	bool equal(float desc[128])
	{
		for (int i = 0; i <= 128; i++)
		{
			if (i == 128)
				return true;
			if (desc[i] != descriptor[i])
				break;
		}
		return false;
	}

};

struct coord {
	float x, y;

	coord() { this->x = -1.f; this->y = -1.f; }
	coord(float x, float y)
	{
		this->x = x; this->y = y;
	}
	//coord& operator= (const coord &coord)
	//{
	//	if (this == &coord)
	//		return *this;
	//	this->x = coord.x;
	//	this->y = coord.y;
	//	return *this;
	//}
};

class stitchImage
{
private:
	cv::String filename[5];
	float scale = 255.f;

	VlSiftFilt **siftFilter;
	
	int numData;

	cv::Mat stitchedImage;
	cv::Mat warpingImage[10];

	cv::Mat* originalData;
	cv::Mat* downSampledData;
	cv::Mat* greyData;
	std::vector<VlSiftKeypoint> *keyPoints;
	vl_sift_pix** siftData;

	std::vector<desc> *descriptors;
	std::vector<matchPair> *matched;

	std::vector<matchPair> *finalMatch;

	cv::Mat homoH[10];

public:
	~stitchImage();
	stitchImage();
	stitchImage(const char** filename, int numData);

	float distError(coord source, coord target, cv::Mat H);
	void extractDesc();
	void match();
	void ransac(int numSample=4);
	void stitch();
	void calcTransform(int src[2], int (&ret)[2], cv::Mat H, bool &found, int sizeX, int sizeY);
public:
	void showSampledData(const char *windowName, cv::Mat *data, int skip = 0, int destroy = 0) const;

}; 