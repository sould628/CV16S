#include "stitchImage.h"

//const dest

stitchImage::stitchImage()
{

}
stitchImage::stitchImage(const char** filename, int numData)
{
	
	this->numData = numData;
	this->originalData = new cv::Mat[numData];
	this->downSampledData = new cv::Mat[numData];
	this->greyData = new cv::Mat[numData];


	for (int i = 0; i < numData; i++)
	{
		this->originalData[i] = cv::imread(filename[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::resize(originalData[i], downSampledData[i], cv::Size(256, 256), 0, 0, CV_INTER_AREA);
		cv::cvtColor(downSampledData[i], greyData[i], CV_RGB2GRAY);
		this->filename[i]=filename[i];
	}

	siftData = new vl_sift_pix*[numData];
	for (int ind = 0; ind < numData; ind++)
	{
		std::cout << "copying Img"<<ind<<std::endl;
		siftData[ind] = new vl_sift_pix[256 * 256];
		for (int i = 0; i < 256; i++)
		{
			for (int j = 0; j < 256; j++)
			{
				cv::Mat data;
//				greyData[ind].convertTo(data, CV_32FC1, 1.f / 255.f);
//				siftData[ind][i*256+j] = data.at<float>(i, j);
				siftData[ind][i * 256 + j] = (float)greyData[ind].at<uchar>(i, j);

			}
		}
	}

	showSampledData("sampleData", downSampledData, 0, 0);
	showSampledData("greyData", greyData, 0, 0);
	siftFilter = new VlSiftFilt*[numData];

	this->keyPoints = new std::vector<VlSiftKeypoint>[numData];
	this->descriptors = new std::vector<desc>[numData];
	this->matched = new std::vector<matchPair>[numData-1];
	this->finalMatch = new std::vector<matchPair>[numData - 1];
}
stitchImage::~stitchImage()
{
	delete[] originalData;
	delete[] downSampledData;
	delete[] greyData;
}





float stitchImage::distError(coord source, coord target, cv::Mat H)
{
	float ha_x, ha_y, ha_w;
	ha_w = source.x * H.at<float>(2, 0) + source.y*H.at<float>(2, 1) + H.at<float>(2, 2);
	ha_x = (source.x * H.at<float>(0,0) + source.y * H.at<float>(0, 1) + H.at<float>(0, 2)) / ha_w;
	ha_y = (source.x * H.at<float>(1, 0) + source.y * H.at<float>(1, 1) + H.at<float>(1, 2)) / ha_w;
	float diff1 = sqrt(pow(ha_x - target.x, 2) + pow(ha_y - target.y, 2));

	cv::Mat invH = H.inv();
	float hb_x, hb_y, hb_w;
	hb_w = target.x * invH.at<float>(2, 0) + target.y * invH.at<float>(2, 1) + invH.at<float>(2, 2);
	hb_x = (target.x * invH.at<float>(0, 0) + target.y * invH.at<float>(0, 1) + invH.at<float>(0, 2)) / hb_w;
	hb_y = (target.x * invH.at<float>(1, 0) + target.y * invH.at<float>(1, 1) + invH.at<float>(1, 2)) / hb_w;
	float diff2 = sqrt(pow(hb_x - source.x, 2) + pow(hb_y - source.y, 2));

	return diff1 + diff2;
}

void stitchImage::extractDesc()
{ 
	int numData = this->numData;
//	double angle[4] = { 0 };
	double angle[4];
	for (int ind = 0; ind < numData; ind++)
	{
		std::cout << "ind: " << ind << std::endl;
		siftFilter[ind] = vl_sift_new(256, 256, log2(256), 3, -1);
		int filterEnd = 0;


		//feature control
		//decrease to eliminate more keypoints
//		vl_sift_set_edge_thresh(siftFilter[ind], 0.2);
		//increase to eliminate more keypoints
//		vl_sift_set_peak_thresh(siftFilter[ind], 1);


		cv::Mat temp;
		downSampledData[ind].copyTo(temp);

		int check = -1;
		while (filterEnd != VL_ERR_EOF)
		{
			if (check == -1)
			{
				filterEnd = vl_sift_process_first_octave(siftFilter[ind], siftData[ind]);
			}
			else {
				filterEnd = vl_sift_process_next_octave(siftFilter[ind]);
			}

			vl_sift_detect(siftFilter[ind]);
			std::cout << "nkeys= " << vl_sift_get_nkeypoints(siftFilter[ind]) << std::endl;
			int nkey = vl_sift_get_nkeypoints(siftFilter[ind]);
			VlSiftKeypoint* key = (VlSiftKeypoint*)vl_sift_get_keypoints(siftFilter[ind]);

			int curOctave = vl_sift_get_octave_index(siftFilter[ind]);
			int curOctaveWidth = vl_sift_get_octave_width(siftFilter[ind]);
			int curOctaveHeight = vl_sift_get_octave_height(siftFilter[ind]);

			vl_sift_pix* octave = vl_sift_get_octave(siftFilter[ind], curOctave);;

			cv::Mat cvoctave = cv::Mat(curOctaveWidth, curOctaveHeight, CV_8UC1);

//			for (int w = 0; w < curOctaveWidth; w++)
//			{
//				for (int h = 0; h < curOctaveHeight; h++)
//				{
//					cvoctave.at<uchar>(w, h) = (uchar)octave[w*curOctaveWidth + h];
//				}
//			}
			check++;
			vl_sift_pix buf[128];


			for (int i = 0; i < nkey; i++)
			{
				int nori = 0;
				cv::Point pt = cv::Point(key[i].ix, key[i].iy);
				cv::Scalar color(0, 0, 1);
//				cv::circle(cvoctave, pt, 2, color);

				cv::circle(temp, pt, 1.f, color);

				nori = vl_sift_calc_keypoint_orientations(siftFilter[ind], angle, &key[i]);
				if (nori == 0)
					std::cout << "error octave mismatch\n";
				for (int j = 0; j < nori; j++)
				{
					vl_sift_calc_keypoint_descriptor(siftFilter[ind], buf, &key[i], angle[j]);
					desc newdesc(buf, &key[i]);
					descriptors[ind].push_back(newdesc);
				}
			}
		}
		displayImage("test", temp, 1, 0);
		cv::imwrite("./result/descExtraction" + filename[ind], temp);
	}

}

void stitchImage::match() {

	for (int ind = 0; ind < numData - 1; ind++)
	{

		VlKDForest *forest = vl_kdforest_new(VL_TYPE_FLOAT, 128, 1, VlDistanceL1);

		float *data1 = new float[128 * (int)descriptors[ind].size()];
		for (int i = 0; i < (int)descriptors[ind].size(); i++)
		{
			for (int k = 0; k < 128; k++)
				data1[128 * i + k] = descriptors[ind][i].descriptor[k];
		}


		vl_kdforest_build(forest, descriptors[ind].size(), data1);

		VlKDForestSearcher *searcher = vl_kdforest_new_searcher(forest);
		VlKDForestNeighbor neighbours[2];


		for (int i = 0; i < (int)descriptors[ind + 1].size(); i++)
		{
			float *data2 = new float[128];
			for (int k = 0; k < 128; k++)
				data2[k] = descriptors[ind + 1][i].descriptor[k];

			int nvisited = vl_kdforestsearcher_query(searcher, neighbours, 2, data2);
			float ratio = neighbours[0].distance / neighbours[1].distance;
			if (ratio < 0.4)
			{
				float desc[128];
				for (int j = 0; j < 128; j++)
				{
					desc[j] = data1[j + neighbours[0].index * 128];
				}
				int index = -1;
				for (int k = 0; k < (int)descriptors[ind].size(); k++)
				{
					if (descriptors[ind][k].equal(desc))
					{
						index = k;
						break;
					}
				}
				if (index != -1)
				{
					VlSiftKeypoint *k1 = descriptors[ind][index].key;
					VlSiftKeypoint *k2 = descriptors[ind + 1][i].key;
					matchPair newMatch(k1, k2);
					matched[ind].push_back(newMatch);
				}
			}
		}
		vl_kdforestsearcher_delete(searcher);
		vl_kdforest_delete(forest);
	}
	std::cout << "match ended\n";

	for (int i = 0; i < numData - 1; i++)
	{
		int numMatched = this->matched[i].size();
		std::cout << "matched pair for img(" << i << ", " << i + 1 << "): " << numMatched << std::endl;
		cv::Mat mat1, mat2, mat3;

		mat1 = this->downSampledData[i]; mat2 = this->downSampledData[i + 1];

		mat3 = cv::Mat(256, 512, CV_8UC3);

		for (int w = 0; w < 256; w++)
		{
			for (int h = 0; h < 256; h++)
			{
				cv::Vec3b val1, val2;
				val1 = mat1.at<cv::Vec3b>(h, w);
				val2 = mat2.at<cv::Vec3b>(h, w);
				mat3.at<cv::Vec3b>(h, w) = val1;
				mat3.at<cv::Vec3b>(h, w + 256) = val2;
			}
		}

		cv::Scalar color(0, 0, 1);

		for (int j = 0; j < numMatched; j++)
		{

			cv::Point pt1_1 = cv::Point((int)matched[i][j].key1->x, (int)matched[i][j].key1->y);
			cv::Point pt1_2 = cv::Point((int)matched[i][j].key2->x+256, (int)matched[i][j].key2->y);

			cv::circle(mat3, pt1_1, 2, color);
			cv::circle(mat3, pt1_2, 2, color);
			cv::line(mat3, pt1_1, pt1_2, color);
		}

		//change 3rd parameter 0 to skip
		displayImage("mat3", mat3, 1, 0);
		cv::imwrite("./result/putativeMatch" + filename[i], mat3);
	}

}

void stitchImage::ransac(int numSample)
{

	for (int i = 0; i < numData - 1; i++)
	{
		homoH[i] = cv::Mat(3, 3, CV_32FC1);
	}

	if (numSample < 4)
	{
		std::cout << "error, sample need to be larger than 4. Enforcing sample to be 4\n";
		numSample = 4;
	}
	srand(time(NULL));
	int index[10];
	coord img1coord[10], img2coord[10];

	float N = INT_MAX;
	int sample_count = 0;


	float t = 1.25; float p = 0.99;


	for (int ind = 0; ind < numData - 1; ind++)
	{
		int highest_inliners = 0;
		sample_count = 0;
		N = INT_MAX;
		while (N > sample_count++)
		{
			if (sample_count % 1000 == 0)
			{
				std::cout << "current iteration: " << sample_count << std::endl;
			}
			int numMatched = matched[ind].size();
			if (matched[ind].size() < 4)
				std::cout << "not enough matches error for index" << ind << std::endl;
			//sample selection
			int numFilled = 0;
			for (int i = 0; i < numSample; i++)
			{
				bool duplicate = false;
				int candidate=(int)rand() % numMatched;;
				for (int j = 0; j <= numFilled; j++)
				{
					if (candidate == index[j])
					{
						duplicate = true;
						break;
					}
					if (i == numFilled - 1)
						break;
				}
				if (!duplicate)
				{
					index[i] = candidate;
					numFilled++;
				}
				else
				{
					i--; continue;
				}
				coord temp1((float)matched[ind][index[i]].key1->x, (float)matched[ind][index[i]].key1->y);
				coord temp2((float)matched[ind][index[i]].key2->x, (float)matched[ind][index[i]].key2->y);


				img1coord[i] = temp1;img2coord[i] = temp2;
			}//selection end


			cv::Mat A = cv::Mat(numSample * 2, 9, CV_32FC1);
			cv::Mat H = cv::Mat(3, 3, CV_32FC1);
			//Getting Homography H
			if (numSample == 4)
			{
				//Construct Matrix A
				for (int i = 0; i < numSample; i++)
				{

					float x1 = img1coord[i].x;
					float y1 = img1coord[i].y;
					float x2 = img2coord[i].x;
					float y2 = img2coord[i].y;

//					A.at<float>(i * 2, 0) = 0;
//					A.at<float>(i * 2, 1) = 0;
//					A.at<float>(i * 2, 2) = 0;
//					A.at<float>(i * 2, 3) = -x1;
//					A.at<float>(i * 2, 4) = -y1;
//					A.at<float>(i * 2, 5) = -1.f;
//					A.at<float>(i * 2, 6) = y2*x1;
//					A.at<float>(i * 2, 7) = y2*x2;
//					A.at<float>(i * 2, 8) = y2;
//
//					A.at<float>(i * 2 + 1, 0) = x1;
//					A.at<float>(i * 2 + 1, 1) = x2;
//					A.at<float>(i * 2 + 1, 2) = 1;
//					A.at<float>(i * 2 + 1, 3) = 0;
//					A.at<float>(i * 2 + 1, 4) = 0;
//					A.at<float>(i * 2 + 1, 5) = 0;
//					A.at<float>(i * 2 + 1, 6) = -x2*x1;
//					A.at<float>(i * 2 + 1, 7) = -x2*y1;
//					A.at<float>(i * 2 + 1, 8) = -x2;
					A.at<float>(i * 2, 0) = x1;
					A.at<float>(i * 2, 1) = y1;
					A.at<float>(i * 2, 2) = 1.f;
					A.at<float>(i * 2, 3) = 0;
					A.at<float>(i * 2, 4) = 0;
					A.at<float>(i * 2, 5) = 0;
					A.at<float>(i * 2, 6) = -x2*x1;
					A.at<float>(i * 2, 7) = -x2*y1;
					A.at<float>(i * 2, 8) = -x2;

					A.at<float>(i * 2+1, 0) = 0;
					A.at<float>(i * 2+1, 1) = 0;
					A.at<float>(i * 2+1, 2) = 0;
					A.at<float>(i * 2+1, 3) = x1;
					A.at<float>(i * 2+1, 4) = y1;
					A.at<float>(i * 2+1, 5) = 1.f;
					A.at<float>(i * 2+1, 6) = -y2*x1;
					A.at<float>(i * 2+1, 7) = -y2*y1;
					A.at<float>(i * 2+1, 8) = -y2;
				}
//				std::cout << "A=\n" << A << std::endl;

				cv::Mat U;
				cv::Mat D;
				cv::Mat Vt;
				cv::Mat V;

				cv::Mat At;
				cv::transpose(A, At);
				cv::Mat AtA = At*A;
				//SVDecomposition
//				cv::SVDecomp(A, D, U, Vt, 4);
//				cv::transpose(Vt, V);

//				cv::SVDecomp(AtA, D, U, Vt, 4);


				cv::Mat h = cv::Mat(9, 1, CV_32FC1);
	
//				h = U.col(U.cols - 1);
//				h = V.col(V.cols-1);
//				h = Vt.col(Vt.cols - 1);

				h/=cv::norm(h);

				cv::Mat Svec;
				cv::SVD::compute(A.t() * A, Svec, U, Vt);
				h = U.col(U.cols - 1);

//				std::cout << "h=\n" << h << std::endl;

				H.at<float>(0, 0) = h.at<float>(0, 0); H.at<float>(0, 1) = h.at<float>(1, 0); H.at<float>(0, 2) = h.at<float>(2, 0);
				H.at<float>(1, 0) = h.at<float>(3, 0); H.at<float>(1, 1) = h.at<float>(4, 0); H.at<float>(1, 2) = h.at<float>(5, 0);
				H.at<float>(2, 0) = h.at<float>(6, 0); H.at<float>(2, 1) = h.at<float>(7, 0); H.at<float>(2, 2) = h.at<float>(8, 0);

			}//H End

			//END DLT

			for (int i = 0; i < numSample; i++)
			{
				float err = distError(img1coord[i], img2coord[i], H);
			}




			cv::Mat xi; cv::Mat xj;
			xi = cv::Mat(3, 1, CV_32FC1);
			xj = cv::Mat(3, 1, CV_32FC1);

			cv::Mat Hinv = H.inv();


			//Actual RANSAC

			int num_inliners = 0;

			float epsilon = 1.f;
			std::vector<matchPair> temp;
			for (int i = 0; i < numMatched; i++)
			{
				coord source, target;
				source.x = matched[ind][i].key1->x;
				source.y = matched[ind][i].key1->y;
				target.x = matched[ind][i].key2->x;
				target.y = matched[ind][i].key2->y;


				float distance = 0.f;
				
				distance = distError(source, target, H);
//				cv::Mat dist1 = xi - Hinv*xj;
//				cv::Mat dist2 = xj - H*xi;
//
//				xi.at<float>(0, 0) = matched[ind][i].key1->x;
//				xi.at<float>(1, 0) = matched[ind][i].key1->y;
//				xi.at<float>(2, 0) = 1.f;
//
//				xj.at<float>(0, 0) = matched[ind][i].key2->x;
//				xj.at<float>(1, 0) = matched[ind][i].key2->y;
//				xj.at<float>(2, 0) = 1.f;
//
//				cv::Vec3f err1 = cv::Vec3f(dist1.at<float>(0, 0), dist1.at<float>(1, 0), dist1.at<float>(2, 0));
//				cv::Vec3f err2 = cv::Vec3f(dist2.at<float>(0, 0), dist2.at<float>(1, 0), dist2.at<float>(2, 0));
//				distance = cv::norm(err1) + cv::norm(err2);

				if (distance < t)
				{
					num_inliners++;
					matchPair newMatch(matched[ind][i].key1, matched[ind][i].key2);
					temp.push_back(newMatch);
				}
			}
			if (highest_inliners < num_inliners)
			{
				highest_inliners = num_inliners;
				homoH[ind] = H;
				std::cout << "ind="<<ind<<std::endl<<"H=\n" << H << std::endl;
				std::cout << "current inliners=" << highest_inliners<<std::endl;
				finalMatch[ind].resize((int)temp.size());
				std::copy(temp.begin(), temp.end(), finalMatch[ind].begin());
			}
			epsilon = 1.f - ((float)num_inliners / (float)numMatched);
			if (epsilon == 1.f)
			{
				N = INT_MAX;
			}
			else 
			{
				N = log(1 - p) / log(1 - powf((1 - epsilon), (float)numSample));
			}

		}//while N ends
	}//for each pair of images ends

	//check finalMatch

	cv::Scalar color(0, 0, 1);
	for (int ind = 0; ind < numData - 1; ind++)
	{
		int numFinalMatch=finalMatch[ind].size();
		std::cout << "numFinal Match of ind " << ind << " is " << numFinalMatch << std::endl;
		cv::Mat temp = cv::Mat(256, 512, CV_8UC1);

		for (int w = 0; w < 256; w++)
		{
			for (int h = 0; h < 256; h++)
			{
				temp.at<uchar>(h, w) = greyData[ind].at<uchar>(h, w);
				temp.at<uchar>(h, w+256) = greyData[ind+1].at<uchar>(h, w);
			}
		}

		for (int i = 0; i < numFinalMatch; i++)
		{
			cv::Point pt1 = cv::Point(finalMatch[ind][i].key1->x, finalMatch[ind][i].key1->y);
			cv::Point pt2 = cv::Point(finalMatch[ind][i].key2->x+256, finalMatch[ind][i].key2->y);

			cv::circle(temp, pt1, 1, color);
			cv::circle(temp, pt2, 1, color);
			cv::line(temp, pt1, pt2, color);
		}

		displayImage("finalMatch", temp, 1, 0);
		cv::imwrite("./result/ransacMatch" + filename[ind], temp);
	}
	//check

	for (int ind = 0; ind < numData - 1; ind++)
	{

	}


	return;
}

void stitchImage::stitch()
{

	int offsetW = 512;
	int offsetH = 232;
	//only for 5 images
	if (numData != 5)
	{
		std::cout << "application only for 5 images\nEnding sequence\n";
			return;
	}
	this->stitchedImage = cv::Mat(720, 1280, CV_8UC3);

	int flag[1280][720] = { 0 };
	//image 3 first
	for (int w = 0; w < 256; w++)
	{
		for (int h = 0; h < 256; h++)
		{
			stitchedImage.at<cv::Vec3b>(232 + h, 512 + w) = downSampledData[2].at<cv::Vec3b>(h, w);
			flag[offsetW + w][offsetH + h] = 1;
		}
	}

	
	cv::Mat homoMat = homoH[0];
	for (int w = 0; w < 1280; w++)
	{
		for (int h = 0; h < 720; h++)
		{
			int src[2] = { w-offsetW, h-offsetH };
			int target[2] = { 0, 0 };
			for (int i = 0; i < numData - 1; i++)
			{
				bool found = false;
				cv::Mat H;
				if (i < 2)
				{
					if (i == 0)
					{
						H = (homoH[0] * homoH[1]).inv();
					}
					else
						H = homoH[i].inv();
				}
				else
				{
					if(i==2)
						H = homoH[i];
					else
					{
						H = homoH[3] * homoH[2];
					}
				}
				calcTransform(src, target, H, found, 256, 256);

				if (found)
				{
					if(i<2)
						stitchedImage.at<cv::Vec3b>(h, w) = downSampledData[i].at<cv::Vec3b>(target[1], target[0]);
					else
						stitchedImage.at<cv::Vec3b>(h, w) = downSampledData[i+1].at<cv::Vec3b>(target[1], target[0]);
				}
			}
		}
	}


	displayImage("stitched", stitchedImage, 1, 0);
	cv::imwrite("./result/stitchedImage", stitchedImage);
}

//src(w,h)
void stitchImage::calcTransform(int src[2], int (&ret)[2], cv::Mat H, bool &found, int sizeX, int sizeY)
{
	ret[0] = ((H.at<float>(0, 0)*src[0]) + (H.at<float>(0, 1)*src[1]) + H.at<float>(0, 2)) / ((H.at<float>(2, 0)*src[0]) + (H.at<float>(2, 1)*src[1]) + H.at<float>(2, 2));
	ret[1] = ((H.at<float>(1, 0)*src[0]) + (H.at<float>(1, 1)*src[1]) + H.at<float>(1, 2)) / ((H.at<float>(2, 0)*src[0]) + (H.at<float>(2, 1)*src[1]) + H.at<float>(2, 2));
	if ((ret[0] >= 0) && (ret[0] < sizeX))
		if ((ret[1] >= 0) && (ret[1] < sizeY))
			found = true;
		else
			found = false;
	else
		found = false;
}

void stitchImage::showSampledData(const char *windowName, cv::Mat *data, int skip, int destroy) const
{
	int numData = this->numData;
	char key = 0;
	int i = 0;
	cv::namedWindow(windowName); cv::namedWindow(windowName, CV_WINDOW_NORMAL);
	while (key != myESC)
	{
		cv::imshow(windowName, data[i]);
		if (skip == 0)
		{
			cv::waitKey(1);
			key = myESC;
		}
		else {
			key = cv::waitKey();
		}
		switch (key)
		{
		case'>':
		case'.':
			i += 1;
			if (i == numData)
				i = 0;
			break;
		case '<':
		case',':
			i -= 1;
			if (i == -1)
				i = numData - 1;
			break;
		}
	}
	if (destroy == 0)
		cv::destroyWindow(windowName);
}