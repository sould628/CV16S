#pragma once
#pragma once

#include <opencv2\opencv.hpp>

class kNNclassifier {
private:
	int numTrainingSet;
	cv::Mat *trainingSet;


public:
	kNNclassifier();
	kNNclassifier(kNNclassifier &kNN);

	inline int getNumTS();
	cv::Mat getTrainingSet(int index) const;

};