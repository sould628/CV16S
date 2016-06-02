#include "KNNCLASSIFIER.H"

kNNclassifier::kNNclassifier()
{
}

kNNclassifier::kNNclassifier(kNNclassifier & kNN)
{
}

inline int kNNclassifier::getNumTS()
{
	return this->numTrainingSet;
}

cv::Mat kNNclassifier::getTrainingSet(int index) const
{
	if (index >= this->numTrainingSet)
	{
		std::cout << "index Number error\n ";
		return cv::Mat::zeros(16, 16, CV_32FC1);
	}
	return cv::Mat(trainingSet[index]);
}
