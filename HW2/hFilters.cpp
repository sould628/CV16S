#include "hFilters.h"

//Const,Dest
hFilters::hFilters()
{
	matType = 0;
	numData = 0;
	dataIndex = 0;
}
hFilters::~hFilters()
{
	delete[] this->testData; testData = 0;
	delete[] this->data;
	this->data = 0;
	std::cout << "object deleted successfully";
};

hFilters::hFilters(char **fileNames, int numData, int MatType)
{
	this->matType = MatType;
	dataIndex = 0;
	this->numData = numData;
	this->data = new cv::Mat[numData];
	this->testData = new cv::Mat[numData];
	cv::Mat temp;
	if (MatType == 0)
	{
		for (int i = 0; i < numData; i++)
		{
			temp = cv::imread((const char*)fileNames[i], CV_LOAD_IMAGE_GRAYSCALE);
			temp.convertTo(this->data[i], CV_32FC1, 1.f / 255.f);
		}
	}
	else
	{
		for (int i = 0; i < numData; i++)
		{
			temp = cv::imread((const char*)fileNames[i], CV_LOAD_IMAGE_COLOR);
			temp.convertTo(this->data[i], CV_32FC3, 1.f/255.f);
		}
	}
	for (int i = 0; i < numData; i++)
	{
		int testX, testY, testWidth, testHeight;
		testX = this->data[i].cols/3; testY = this->data[i].rows/3;
		testWidth = 120; testHeight = 60;
		cv::Rect ROI(testX, testY, testWidth, testHeight);
		this->data[i](ROI).copyTo(testData[i]);
	}
}

void hFilters::loadImageRGBtoData(char **fileNames, int numData, int indBegin, int indEnd)
{
	this->numData = numData * 3;
	this->matType = 0;
	cv::Mat temp;
	cv::Mat channel[3];
	data = new cv::Mat[numData * 3];
	cv::Mat t;

	for (int i = 0; i < numData; i++)
	{
		temp = cv::imread((const char*)fileNames[i+indBegin], CV_LOAD_IMAGE_COLOR);
		cv::split(temp, channel);
		t = cv::Mat::zeros(temp.rows, temp.cols, CV_32FC1);
		for (int j = 0; j < 3; j++)
		{
			channel[j].convertTo(t, CV_32FC1, 1/255.f);
			data[i * 3 + j] = t.clone();
		}
	}

}


//Median Filter Functions
cv::Mat hFilters::weightedMedian(int input, int kType, int dataUsed, int kernelSizeX, int kernelSizeY, 
	float sigmaX, float sigmaY, int guidance, float e) {
	
	cv::Mat tInput;
	if (this->matType == 0) {
		if (dataUsed == 0)	this->data[input].convertTo(tInput, CV_8UC1, 255);
		else if (dataUsed == 1)	this->testData[input].convertTo(tInput, CV_8UC1, 255);
	}
	else if (this->matType == 1)
	{
		if (dataUsed == 0)	this->data[input].convertTo(tInput, CV_8UC3, 255);
		else if (dataUsed == 1)	this->testData[input].convertTo(tInput, CV_8UC3, 255);
	}
	int matr, matc;
	matr = tInput.rows; matc = tInput.cols;

	//create histogram map
	cv::Mat map[256];
	for (int k = 0; k < 256; k++)
	{
		map[k] = cv::Mat::zeros(matr, matc, CV_32FC1);
		for (int i = 0; i < matr; i++)
		{
			for (int j = 0; j < matc; j++)
			{
				uchar pixelValue = tInput.at<uchar>(i, j);
				if (pixelValue == (int)k)
					map[k].at<float>(i, j) = 1.f;
				else
					map[k].at<float>(i, j) = 0.f;
			}
		}
	}

	cv::Size kernelSize(kernelSizeX, kernelSizeY);
	float kernelArea = (float)kernelSizeX*kernelSizeY;


	cv::Mat output=cv::Mat(matr, matc, CV_8UC1);
	cv::Mat bimap[256]; cv::Mat boxmap[256]; cv::Mat gmap;
	cv::Mat guidmap[256]; cv::Mat guidfilmap[256];
	//basic box
	if (kType == fil::kernel::box)
	{
		//filter on the maps f(x,i)
		for (int k = 0; k < 256; k++)
		{
			cv::boxFilter(map[k], boxmap[k], -1, kernelSize);
		}
		float curArea = 0;
		for (int i = 0; i < matr; i++)
		{
			for (int j = 0; j < matc; j++)
			{
				for (int k = 0; k < 256; k++)
				{
					curArea += boxmap[k].at<float>(i, j);
					if (curArea >= 0.5)
					{
						output.at<uchar>(i, j) = k;
						break;
					}
				}
				curArea = 0;
			}
		}
	}
	//bilateral filter
	else if (kType == fil::kernel::bilateral)
	{
		int diameter = kernelSizeX;
		float sigmaColor = sigmaX; float sigmaSpace = sigmaY;
		for (int k = 0; k < 256; k++)
		{
			cv::bilateralFilter(map[k], bimap[k], diameter, sigmaColor, sigmaSpace);
		}
		float curArea = 0;
		for (int i = 0; i < matr; i++)
		{
			for (int j = 0; j < matc; j++)
			{
				for (int k = 0; k < 256; k++)
				{
					curArea += bimap[k].at<float>(i, j);
					if (curArea >= 0.5)
					{
						output.at<uchar>(i, j) = k;
						break;
					}
				}
				curArea = 0;
			}
		}
	}
	else if (kType == fil::kernel::guided)
	{
		if (guidance == -1)
		{
			std::cout << "guidance image index error" << std::endl;
			return cv::Mat::zeros(10, 10, CV_32FC1);
		}
		if (this->matType == 0) {
			if (dataUsed == 0)	this->data[guidance].convertTo(gmap, CV_8UC1, 255);
			else if (dataUsed == 1)	this->testData[guidance].convertTo(gmap, CV_8UC1, 255);
		}
		//histogram for guidance
		for (int k = 0; k < 256; k++)
		{
			guidmap[k] = cv::Mat::zeros(matr, matc, CV_32FC1);
			for (int i = 0; i < matr; i++)
			{
				for (int j = 0; j < matc; j++)
				{
					uchar pixelValue = gmap.at<uchar>(i, j);
					if (pixelValue == (int)k)
						guidmap[k].at<float>(i, j) = 1.f;
					else
						guidmap[k].at<float>(i, j) = 0.f;
				}
			}
		}
		//histogram for guidance end
		for (int k = 0; k < 256; k++)
		{
			guidfilmap[k]=this->GuidedFilterG(map[k], guidmap[k], kernelSizeX, kernelSizeY, sigmaX, sigmaY, e);
		}
		float curArea = 0;
		for (int i = 0; i < matr; i++)
		{
			for (int j = 0; j < matc; j++)
			{
				for (int k = 0; k < 256; k++)
				{
					curArea += guidfilmap[k].at<float>(i, j);
					if (curArea >= 0.5)
					{
						output.at<uchar>(i, j) = k;
						break;
					}
				}
				curArea = 0;
			}
		}

	}



	return output;
}

//Guided Filter Functions
cv::Mat hFilters::GuidedFilterG(cv::Mat input, cv::Mat guidance,
	int kernelSizeX, int kernelSizeY, float sigmaX, float sigmaY, float e)
{
	cv::Mat p = input;
	cv::Mat guid = guidance;
	int pr, pc, gr, gc;
	pr = p.rows; pc = p.cols; gr = guid.rows; gc = guid.cols;
	if ((pr != gr) || (pc != gc))
	{
		std::cout << "input image and guid image dimension mismatch" << std::endl;
		if (this->matType == 0)
			return cv::Mat::zeros(1, 1, CV_32FC1);
		else
			return cv::Mat::zeros(1, 1, CV_32FC3);
	}
	cv::Mat mean_guid, mean_p, corr_guid, corr_gp, var_guid, cov_gp, a, b, mean_a, mean_b, q;
	cv::Size kSize(kernelSizeX, kernelSizeY);
	if (this->matType == 0)
	{
		//1
		cv::Mat temp = cv::Mat(p.rows, p.cols, CV_32FC1);
		cv::GaussianBlur(guid, mean_guid, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(p, mean_p, kSize, sigmaX, sigmaY);
		temp = elementMul(guid, guid);
		cv::GaussianBlur(temp, corr_guid, kSize, sigmaX, sigmaY);
		temp = elementMul(guid, p);
		cv::GaussianBlur(temp, corr_gp, kSize, sigmaX, sigmaY);

		//2
		var_guid = corr_guid - elementMul(mean_guid, mean_guid);
		cov_gp = corr_gp - elementMul(mean_guid, mean_p);

		//3
		a = elementDiv(cov_gp, (var_guid + e));
		b = mean_p - elementMul(a, mean_guid);

		//4
		cv::GaussianBlur(a, mean_a, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(b, mean_b, kSize, sigmaX, sigmaY);

		//5
		q = elementMul(mean_a, guid) + mean_b;
	}
	else
	{
		//1
		cv::Mat temp = cv::Mat(p.rows, p.cols, CV_32FC3);
		cv::GaussianBlur(guid, mean_guid, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(p, mean_p, kSize, sigmaX, sigmaY);
		temp = elementMul(guid, guid, this->matType);
		cv::GaussianBlur(temp, corr_guid, kSize, sigmaX, sigmaY);
		temp = elementMul(guid, p, this->matType);
		cv::GaussianBlur(temp, corr_gp, kSize, sigmaX, sigmaY);

		//2
		var_guid = corr_guid - elementMul(mean_guid, mean_guid, this->matType);
		cov_gp = corr_gp - elementMul(mean_guid, mean_p, this->matType);

		//3
		a = elementDiv(cov_gp, (var_guid + e), this->matType);
		b = mean_p - elementMul(a, mean_guid, this->matType);

		//4
		cv::GaussianBlur(a, mean_a, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(b, mean_b, kSize, sigmaX, sigmaY);

		//5
		q = elementMul(mean_a, guid, this->matType) + mean_b;
	}


	//	return q;
	return q;
}

cv::Mat hFilters::GuidedFilterG(int input, int guidance,
	int kernelSizeX, int kernelSizeY, float sigmaX, float sigmaY, float e)
{
	cv::Mat p = this->data[input];
	cv::Mat guid = this->data[guidance];
	int pr, pc, gr, gc;
	pr = p.rows; pc = p.cols; gr = guid.rows; gc = guid.cols;
	if ((pr != gr) || (pc != gc))
	{
		std::cout << "input image and guid image dimension mismatch" << std::endl;
		if (this->matType == 0)
			return cv::Mat::zeros(1, 1, CV_32FC1);
		else
			return cv::Mat::zeros(1, 1, CV_32FC3);
	}
	cv::Mat mean_guid, mean_p, corr_guid, corr_gp, var_guid, cov_gp, a, b, mean_a, mean_b, q;
	cv::Size kSize(kernelSizeX, kernelSizeY);
	if (this->matType == 0)
	{
		//1
		cv::Mat temp = cv::Mat(p.rows, p.cols, CV_32FC1);
		cv::GaussianBlur(guid, mean_guid, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(p, mean_p, kSize, sigmaX, sigmaY);
		temp=elementMul(guid, guid);
		cv::GaussianBlur(temp, corr_guid, kSize, sigmaX, sigmaY);
		temp=elementMul(guid, p);
		cv::GaussianBlur(temp, corr_gp, kSize, sigmaX, sigmaY);

		//2
		var_guid = corr_guid-elementMul(mean_guid, mean_guid);
		cov_gp = corr_gp-elementMul(mean_guid, mean_p);

		//3
		a = elementDiv(cov_gp, (var_guid + e));
		b = mean_p - elementMul(a, mean_guid);

		//4
		cv::GaussianBlur(a, mean_a, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(b, mean_b, kSize, sigmaX, sigmaY);

		//5
		q = elementMul(mean_a, guid) + mean_b;
	}
	else
	{
		//1
		cv::Mat temp = cv::Mat(p.rows, p.cols, CV_32FC3);
		cv::GaussianBlur(guid, mean_guid, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(p, mean_p, kSize, sigmaX, sigmaY);
		temp = elementMul(guid, guid, this->matType);
		cv::GaussianBlur(temp, corr_guid, kSize, sigmaX, sigmaY);
		temp = elementMul(guid, p, this->matType);
		cv::GaussianBlur(temp, corr_gp, kSize, sigmaX, sigmaY);

		//2
		var_guid = corr_guid - elementMul(mean_guid, mean_guid, this->matType);
		cov_gp = corr_gp - elementMul(mean_guid, mean_p, this->matType);

		//3
		a = elementDiv(cov_gp, (var_guid + e), this->matType);
		b = mean_p - elementMul(a, mean_guid, this->matType);

		//4
		cv::GaussianBlur(a, mean_a, kSize, sigmaX, sigmaY);
		cv::GaussianBlur(b, mean_b, kSize, sigmaX, sigmaY);

		//5
		q = elementMul(mean_a, guid, this->matType) + mean_b;
	}

	return q;
}



//display Functions
void hFilters::displayImage (const char *windowName, int index, int skip, int flag, int display)
{
	this->windowList.push_back(windowName);
	char key=0;
	this->dataIndex = index;
	cv::namedWindow(windowName); cv::namedWindow(windowName, CV_WINDOW_NORMAL);

	while (key != myESC)
	{
		if(display==0)
			cv::imshow(windowName, this->data[this->dataIndex]);
		else if (display==1)
			cv::imshow(windowName, this->testData[this->dataIndex]);
		if (skip == 0)
		{
			cv::waitKey(1);
			key = myESC;
		}
		else {
			key = cv::waitKey();

			if (key == myNext)
				this->nextIndex();
			else if (key == myPrev)
				this->prevIndex();
		}

	}
	if (flag == 0)
		this->destroyWindow(windowName);

}
void hFilters::destroyWindow(const char*windowName)
{
	std::vector <const char*>::const_iterator pos;
	pos = (std::find(this->windowList.begin(), this->windowList.end(), windowName));
	if (pos > this->windowList.end())
		std::cout << "No window with the name exist" << std::endl;
	else
	{
		cv::destroyWindow(windowName);
		this->windowList.erase(pos);
	}
}

//PSNR function
float fil::PSNR(cv::Mat clean, cv::Mat filtered)
{
	int cleanType, filteredType;
	cleanType = clean.type(); filteredType = filtered.type();
	if (cleanType != filteredType)
	{
		std::cout << "matrix type mismatch error" << std::endl;
		return -1;
	}
	int cleanr, cleanc, filr, filc;
	cleanr = clean.rows; cleanc = clean.cols;
	filr = filtered.rows; filc = filtered.cols;
	float MSE = 10.f; float output = 1.f;
	if ((cleanr != filr) || (cleanc != filc))
	{
		std::cout << "Matrix size mismatch error" << std::endl;
		return -1;
	}
	float peak = 0.f;
	for (int i = 0; i < filr; i++)
	{
		for (int j = 0; j < filc; j++)
		{
			if (cleanType == CV_32FC1)
			{
				float cleanPix = clean.at<float>(i, j);
				float filteredPix = filtered.at<float>(i, j);

				if (filteredPix > peak)
					peak = filteredPix;

				float error = cleanPix-filteredPix;
				MSE += error*error;
			}
			else if (cleanType == CV_32FC3)
			{

			}
			else if (cleanType == CV_8U)
			{

			}
		}
	}
	MSE /= (filr*filc);
	if (MSE == 0.f)
	{
		std::cout << "no noise exists";
		return 0;
	}
	output = 20.f*log10f(peak / sqrtf(MSE));

	return output;
}
