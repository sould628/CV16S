#include <iostream>
#include <math.h>
#include <GL/glut.h>

#include <opencv2\opencv.hpp>

#include "FreeImage.h" //for reading Data

#include "Camera.h"

#include "glovar.h"




#define sysPause system("pause>nul")

using namespace cv;

FIBITMAP* LoadImage(const char* filename, int &imageWidth, int &imageHeight) {
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename);
	RGBQUAD pixel;
	if (format == -1)
	{
		std::cout << "Could not find image: " << filename << " - Aborting." << std::endl;
		sysPause;
			exit(-1);
	}
	if (format == FIF_UNKNOWN)
	{
		std::cout << "Couldn't determine file format - attempting to get from file extension..." << std::endl;

		format = FreeImage_GetFIFFromFilename(filename);

		if (!FreeImage_FIFSupportsReading(format))
		{
			std::cout << "Detected image format cannot be read!" << std::endl;
			sysPause;
			exit(-1);
		}
	}
	FIBITMAP* bitmap = FreeImage_Load(format, filename);
	int bitsPerPixel = FreeImage_GetBPP(bitmap);

	FIBITMAP* bitmap32;

	if (bitsPerPixel == 32)
	{
		std::cout << "Source image has " << bitsPerPixel << " bits per pixel. Skipping conversion." << std::endl;
		bitmap32 = bitmap;
	}
	else
	{
		std::cout << "Source image has " << bitsPerPixel << " bits per pixel. Converting to 32-bit colour." << std::endl;
		bitmap32 = FreeImage_ConvertTo32Bits(bitmap);
	}

	imageWidth = FreeImage_GetWidth(bitmap32);
	imageHeight = FreeImage_GetHeight(bitmap32);

	FreeImage_GetPixelColor(bitmap32, 25, 15, &pixel);
	float r, g, b;
	r = pixel.rgbRed;
	b = pixel.rgbBlue;
	g = pixel.rgbGreen;
	std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "." << std::endl;

	return bitmap32;

}


void initSharedMemory() 
{
	float cinTemp;
	std::cout << "Setting Parameters:" << std::endl << "Set number of Data to be used (must be in between 3 to 11, Larger the Better): ";
	std::cin >> numImagesUsed;
	std::cout << "Set Lambda to be used (0 for default value=2): ";
	std::cin >> cinTemp; if (cinTemp != 0) lambda = cinTemp;
	std::cout << "Set mu to be used (0 for default value=1/imgWidth): ";
	std::cin >> cinTemp; if (cinTemp != 0) mu = cinTemp;
	std::cout << "Set nu to be used (0 for default value=1/imgHeight): ";
	std::cin >> cinTemp; if (cinTemp != 0) nu = cinTemp;
	
	curIndex = numData * 100;
	index = curIndex%numData;
	rawData = new FIBITMAP*[numData];
	imageWidth = new int[numData]; imageHeight = new int[numData];
	textureData = new GLubyte*[numData];
}

void keyboardCB(unsigned char key, int x, int y)
{

	switch (key)
	{
	case '`':
	case '~':
	{
		renderMode = 0;
		std::cout << "\ncurrent Rendering RawData" << std::endl;
		break;
	}
	case '<':
	case ',':
	{
		index=--curIndex%numData;
		std::cout << "current Showing Picture: " << index + 1 << std::endl;
		break;
	}
	case '>':
	case '.':
	{
		index = ++curIndex%numData;
		std::cout << "current Showing Picture: " << index + 1 << std::endl;
		break;
	}
	}
	
	glutPostRedisplay();
}

void displayCB() 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
//	gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);

	cam->glRender();


	switch (renderMode)
	{
	case 0:
	{
		glDrawPixels(imageWidth[index], imageHeight[index], GL_RGBA, GL_UNSIGNED_BYTE, textureData[index]);
		break;
	}
	case 1:
	{
//		glBegin(GL_QUADS);
//		for (int j = 0; j < imgHeight-1; j++)
//			for (int i = 0; i < imgWidth-1; i++)
//			{
//				{
//					glColor3f(
//						(1-(pointCloud[imgHeight*j+i][2] - pc_min)*3 / pc_max_min), 
//						(2-(pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min),
//						(3-(pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min));
//					glVertex3f(pointCloud[imgHeight*j + i][0], pointCloud[imgHeight*j + i][1], pointCloud[imgHeight*j + i][2]);
//					glVertex3f(pointCloud[imgHeight*j + i+1][0], pointCloud[imgHeight*j + i + 1][1], pointCloud[imgHeight*j + i + 1][2]);
//					glVertex3f(pointCloud[imgHeight*(j+1) + i][0], pointCloud[imgHeight*(j + 1) + i][1], pointCloud[imgHeight*(j + 1) + i][2]);
//					glVertex3f(pointCloud[imgHeight*(j+1) + i+1][0], pointCloud[imgHeight*(j + 1) + i + 1][1], pointCloud[imgHeight*(j + 1) + i + 1][2]);
//				}
//
//			}
//		glEnd();
		glShadeModel(GL_SMOOTH);
		glBegin(GL_POINTS);
		for (int j = 0; j < imgHeight; j++)
			for (int i = 0; i < imgWidth; i++)
			{
				{
					if (((pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min) < (1.f / 3.f))
						glColor3f(0, (pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min, 0);
					else if (((pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min) < (2.f / 3.f))
						glColor3f(0, 0, (pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min);
					else
						glColor3f((pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min, 0, 0);
//					glColor3f(
//						(1 - (pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min),
//						(3 - (pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min),
//						((pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min)+1.5);
					glVertex3f(pointCloud[imgHeight*j + i][0], pointCloud[imgHeight*j + i][1], pointCloud[imgHeight*j + i][2]);
				}
			}
		glEnd();
//		glBegin(GL_LINES);
//		for (int j = 0; j < imgHeight; j++)
//			for (int i = 0; i < imgWidth; i++)
//			{
//				{
//					glColor3f(0,0,0);
//					glVertex3f(pointCloud[imgHeight*j + i][0], pointCloud[imgHeight*j + i][1], 0);
//					if (((pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min) < (1.f / 3.f))
//						glColor3f(0, 1, 0);
//					else if (((pointCloud[imgHeight*j + i][2] - pc_min) / pc_max_min) < (2.f / 3.f))
//						glColor3f(0, 0, 1);
//					else
//						glColor3f(1, 0, 0);
//					glVertex3f(pointCloud[imgHeight*j + i][0], pointCloud[imgHeight*j + i][1], pointCloud[imgHeight*j + i][2]);
//				}
//			}
//		glEnd();
		break;
	}
	case 2:
	{
		glBegin(GL_POINTS);
		for (int j = 0; j < imgHeight; j++)
			for (int i = 0; i < imgWidth; i++)
			{
				{
					if (((basPointCloud[imgHeight*j + i][2] - basPc_min) / basPc_max_min) < (1.f / 3.f))
						glColor3f(0, (basPointCloud[imgHeight*j + i][2] - basPc_min) / pc_max_min, 0);
					else if (((basPointCloud[imgHeight*j + i][2] - basPc_min) / basPc_max_min) < (2.f / 3.f))
						glColor3f(0, 0, (basPointCloud[imgHeight*j + i][2] - basPc_min) / pc_max_min);
					else
						glColor3f((basPointCloud[imgHeight*j + i][2] - basPc_min) / basPc_max_min, 0, 0);
					//					glColor3f(
					//						(1 - (pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min),
					//						(3 - (pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min),
					//						((pointCloud[imgHeight*j + i][2] - pc_min) * 3 / pc_max_min)+1.5);
					glVertex3f(basPointCloud[imgHeight*j + i][0], basPointCloud[imgHeight*j + i][1], basPointCloud[imgHeight*j + i][2]);
				}
			}
		glEnd();
		break;
	}
	}
	glFlush();
}
void mouseCB(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		click_pos[0] = (x - windowWidth / 2.0f) / (windowWidth / 2.0f);
		click_pos[1] = (y - windowHeight / 2.0f) / (windowHeight / 2.0f);
	}
	else if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) {
		click_pos[0] = (x - windowWidth / 2.0f) / (windowWidth / 2.0f);
		click_pos[1] = (y - windowHeight / 2.0f) / (windowHeight / 2.0f);
	}
	button_id = button;
	glutPostRedisplay();
}
void motionCB(int x, int y) {
	GLfloat present[2];

	switch (button_id) {
	case GLUT_LEFT_BUTTON:
		present[0] = (GLfloat)(x - windowWidth / 2.0f) / (GLfloat)(windowWidth / 2.0f);
		present[1] = (GLfloat)(y - windowHeight / 2.0f) / (GLfloat)(windowHeight / 2.0f);

		cam->trackball(click_pos, present);

		click_pos[0] = present[0];
		click_pos[1] = present[1];
		break;
	case GLUT_MIDDLE_BUTTON:
		present[0] = (GLfloat)(x - windowWidth / 2.0f) / (GLfloat)(windowWidth / 2.0f);
		present[1] = (GLfloat)(y - windowHeight / 2.0f) / (GLfloat)(windowHeight / 2.0f);
		if (present[1] - click_pos[1] < 0)
			cam->dollyin();
		else if (present[1] - click_pos[1] > 0)
			cam->dollyout();

		click_pos[0] = present[0];
		click_pos[1] = present[1];
		break;
	}
	glutPostRedisplay();
}
void mainMenu(int value) {
	switch (value)
	{
	case 0:
		renderMode = 1;
		break;
	case 1:
		renderMode = 2;
		break;
	}
	glutPostRedisplay();
}

void initGLUT(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("CV2016-HW1");
	glutDisplayFunc(displayCB);
	glutKeyboardFunc(keyboardCB);
	glutMouseFunc(mouseCB);
	glutMotionFunc(motionCB);
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Calibrated", 0);
	glutAddMenuEntry("Uncalibrated", 1);
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	//Textures

}

cv::Mat globalHeights(cv::Mat Pgrads, cv::Mat Qgrads) {

	cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));

	float lambda = 1.0f;
	float mu = 1.0f;

	cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
	for (int i = 0; i<Pgrads.rows; i++) {
		for (int j = 0; j<Pgrads.cols; j++) {
			if (i != 0 || j != 0) {
				float u = sin((float)(i * 2 * CV_PI / Pgrads.rows));
				float v = sin((float)(j * 2 * CV_PI / Pgrads.cols));

				float uv = pow(u, 2) + pow(v, 2);
				float d = (1.0f + lambda)*uv + mu*pow(uv, 2);
				Z.at<cv::Vec2f>(i, j)[0] = (u*P.at<cv::Vec2f>(i, j)[1] + v*Q.at<cv::Vec2f>(i, j)[1]) / d;
				Z.at<cv::Vec2f>(i, j)[1] = (-u*P.at<cv::Vec2f>(i, j)[0] - v*Q.at<cv::Vec2f>(i, j)[0]) / d;
			}
		}
	}

	/* setting unknown average height to zero */
	Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
	Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

	cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

	return Z;
}



int main(int argc, char** argv)
{
	initSharedMemory();
	char** nDataArray;
	nDataArray = new char*[numData];
	nDataArray[0] = nData0; nDataArray[1] = nData1; nDataArray[2] = nData2; nDataArray[3] = nData3; nDataArray[4] = nData4; nDataArray[5] = nData5;
	nDataArray[6] = nData6; nDataArray[7] = nData7; nDataArray[8] = nData8; nDataArray[9] = nData9; nDataArray[10] = nData10; nDataArray[11] = nData11;


#pragma region opencv
	int CVcurIndex = numData * 100 + 1;
	int CVindex = CVcurIndex%numData;
	std::cout << "opencv version: " << CV_VERSION << std::endl;
	bool bCVInProcess = true;
	namedWindow("cv_window");
	namedWindow("cv_window", CV_WINDOW_NORMAL);
	cv::Mat* ambExtractImage = new Mat[numData];
	//Image Loading
	image = new Mat[numData]; //one for ambient
	cropImage = new Mat[numData];
	for (int i = 0; i < numData; i++)
	{
		image[i] = imread((const char*)nDataArray[i], CV_LOAD_IMAGE_UNCHANGED);
	}
	//Image Processing

	//1. Removing Ambient from original images
	for (int i = 0; i < numData; i++)
	{
		ambExtractImage[i] = image[i] - image[0];
		ambExtractImage[i] = max(ambExtractImage[i], 0);
	}

	//2. Crop face
	//method1
	cv::CascadeClassifier face_classifier;
	bool loaded = face_classifier.load(frontal_face_classifier_path);


	std::vector<cv::Rect> *faces;
	faces = new std::vector<cv::Rect>[numData];
	for (int i = 0; i < numData; i++)
		face_classifier.detectMultiScale(ambExtractImage[i], faces[i], 1.1, 3,
			CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

	////////////////
	//3. cropping
	/////////////////////
	bool showFaces = true;
	bool showExtracted = true;
	cv::Mat *temp1, *temp2;
	temp1 = new Mat[numData];
	temp2 = new Mat[numData];

	//method 1//////////
	for (int j = 0; j < numData; j++)
	{
		temp1[j] = ambExtractImage[j].clone();
		temp2[j] = image[j].clone();;
		for (int i = 0; i < faces[j].size(); i++)
		{
			//			std::cout << "\ncurrent Image: " << CVindex << std::endl;
			cv::Point lb(faces[j][i].x + faces[j][i].width,
				faces[j][i].y + faces[j][i].height);
			cv::Point tr(faces[j][i].x, faces[j][i].y);

			cv::rectangle(temp1[j], lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
			cv::rectangle(temp1[j], lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
			cv::rectangle(temp2[j], lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
			cv::rectangle(temp2[j], lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);

		}
	}


	std::cout << "\nNavigte Images with ',' '.'.\nSelect Current Crop Region by Pressing Space.\n";
	//Image Loop (select cropping region)
	//Showing Image Loaded
	while (bCVInProcess)
	{
		int cvImageWidth, cvImageHeight;
		if (showFaces)
		{
			if (!showExtracted)
				imshow("cv_window", temp2[CVindex]);
			else
				imshow("cv_window", temp1[CVindex]);
		}
		else
		{
			if (!showExtracted)
				imshow("cv_window", image[CVindex]);
			else
				imshow("cv_window", ambExtractImage[CVindex]);
		}


		char ch = waitKey();

		if (ch == 27) break; //ESC
		if (ch == 32) //SPACE
		{
			showFaces = true;
			showExtracted = true;
			cv::Point lb(faces[CVindex][0].x + faces[CVindex][0].width,
				faces[CVindex][0].y + faces[CVindex][0].height);
			cv::Point tr(faces[CVindex][0].x, faces[CVindex][0].y);

			std::cout << "Croping Face based on current Image\n";
			for (int i = 0; i < numData - 1; i++)
			{
				cropImage[i] = ambExtractImage[i + 1](cv::Rect(lb, tr));
			}
			cropImage[numData - 1] = ambExtractImage[0](cv::Rect(lb, tr));

			std::cout << "Croping Face Finished\n Proceeding to Next Step\n";
			CVcurIndex = numData * 100;
			break;
		}
		if (ch == 'f') { showFaces ? (showFaces = false) : (showFaces = true); }
		if (ch == '`') { showExtracted ? (showExtracted = false) : (showExtracted = true); }
		if (ch == ',') { CVcurIndex--; CVindex = CVcurIndex%numData; }
		if (ch == '.') { CVcurIndex++; CVindex = CVcurIndex%numData; }
	}


	destroyWindow("cv_window");
//	destroyAllWindows();

	namedWindow("faceImage"); namedWindow("faceImage", CV_WINDOW_NORMAL);
	namedWindow("surfaceNormal");namedWindow("surfaceNormal", CV_WINDOW_NORMAL);
	namedWindow("p-map");namedWindow("p-map", CV_WINDOW_NORMAL);
	namedWindow("q-map");namedWindow("q-map", CV_WINDOW_NORMAL);
	namedWindow("albedo");namedWindow("albedo", CV_WINDOW_NORMAL);
	namedWindow("z-map"); namedWindow("z-map", CV_WINDOW_NORMAL);
	namedWindow("simulation"); namedWindow("simulation", CV_WINDOW_NORMAL);

	CVcurIndex = numData * 100+1;
	CVindex = CVcurIndex%numData;

	//HW1-1 Photometric Stereo
	//step1. Calculate incident vectors

	cv::Mat lightMatrix(numImagesUsed,3,CV_32FC1);

	cv::Vec3f *lightDir;
	lightDir = new cv::Vec3f[numData-1];
	for (int i = 0; i < numData - 1; i++)
	{
		float x, y, z;
		float phi = elevation[i + 1] * mPI / 180.f;
		float theta = azimuth[i + 1] * mPI / 180.f;
		x = cos(phi)*sin(theta);
		y = sin(phi);
		z = cos(phi)*cos(theta);
		lightDir[i] = cv::Vec3f(x, y, z);
		lightDir[i] = lightDir[i] / norm(lightDir[i]);
		float checker;
		checker=lightDir[i][0] * lightDir[i][0] + lightDir[i][1] * lightDir[i][1] + lightDir[i][2] * lightDir[i][2];
		std::cout << "";
	}

	//create Light Matrix

	int ind[3] = {0, 1, 2};

	for (int i = 0; i < numImagesUsed; i++) 
	{
		for (int j = 0; j < 3; j++)
		{
//			lightMatrix.row(i).col(j) = lightDir[i].val[j];
			lightMatrix.at<float>(i, j) = lightDir[i].val[j];
			std::cout << "";
		}
	}
	std::cout << "lightMatrix= " << lightMatrix << std::endl;


	cv::Mat inverseLightMatrix = lightMatrix.inv(DECOMP_SVD);

	std::cout << "inverse = \n" << inverseLightMatrix << std::endl;


	std::cout << "inverse*light=\n" << inverseLightMatrix*lightMatrix << std::endl;
	cv::Mat *convCropImage;
	convCropImage = new Mat[numData - 1];
	for (int i = 0; i < numData - 1; i++)
	{
		cropImage[i].convertTo(convCropImage[i], CV_32FC1, 1.f / 255.f);
	}


	//STEP2. Calculate Surface Normals
	//Method1

	imgWidth = cropImage[0].rows; imgHeight = cropImage[0].cols;
	std::cout << "faceImage (w,h) = (" << imgWidth << ", " << imgHeight << ")\n";

	//Mat of p , q, albedo, surfaceNormals

	cv::Mat p_map(imgHeight, imgWidth, CV_32FC1);
	cv::Mat q_map(imgHeight, imgWidth, CV_32FC1);
	cv::Mat *simulationMap;
	simulationMap = new cv::Mat[numData-1];
//	cv::Mat albedo_map(imgWidth, imgHeight, CV_32FC1);

	cv::Mat surfaceNormal(imgHeight, imgWidth, CV_32FC3);
	cv::Mat reflectanceFactor(imgHeight, imgWidth, CV_32FC1);


	///////////////////////////////////////////////////////////////////////////
	//Calculate Maps
	////////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < imgHeight; i++)
	{
		for (int j = 0; j < imgWidth; j++)
		{
			cv::Mat aux(3, 1, CV_32FC1);
			
			cv::Vec3f aux2;
			cv::Vec3f intensity2;
			cv::Mat matForm;

			//Number Of Images=3 Calculate b
			if (numImagesUsed == 3)
			{
				for (int ii = 0; ii < numImagesUsed; ii++)
				{
					intensity2[ii] = convCropImage[ii].at<float>(i, j);
				}
				matForm = inverseLightMatrix*Mat(intensity2);
				aux2[0] = matForm.at<float>(0, 0); aux2[1] = matForm.at<float>(1, 0); aux2[2] = matForm.at<float>(2, 0);
			}
			//numImages USed>3
			else
			{
				cv::Mat intensity(numImagesUsed, 1, CV_32FC1);
				for (int ii = 0; ii < numImagesUsed; ii++)
				{
					//intensity.at<float>(ii, 0) = convCropImage[ii].at<float>(i, j);
					intensity.at<float>(ii, 0) = convCropImage[ii].at<float>(i, j);
				}

				matForm = ((lightMatrix.t()*lightMatrix).inv(DECOMP_SVD))*(lightMatrix.t())*intensity;

				aux2[0] = matForm.at<float>(0, 0); aux2[1] = matForm.at<float>(1, 0); aux2[2] = matForm.at<float>(2, 0);
				std::cout << "";
			}
			
			float nx, ny, nz;
			float sigma=norm(aux2);
			reflectanceFactor.at<float>(i,j) = sigma;
			surfaceNormal.at<cv::Vec3f>(i,j)=(1.f/sigma)*aux2;
			nx = surfaceNormal.at<cv::Vec3f>(i, j)[0], ny = surfaceNormal.at<cv::Vec3f>(i, j)[1], nz = surfaceNormal.at<cv::Vec3f>(i, j)[2];
			p_map.at<float>(i, j) = -(nx / nz); q_map.at<float>(i, j) = -(ny / nz);


		}
	}



//	//BasRelief Method (SVDecomposition)


	cv::Mat basIntensity(imgWidth*imgHeight, numImagesUsed, CV_32FC1);
	for (int k = 0; k < numImagesUsed; k++)
	{
		for (int j = 0; j < imgHeight; j++)
		{
			for (int i = 0; i < imgWidth; i++)
			{
				basIntensity.at<float>(imgHeight*j + i, k) = convCropImage[k].at<float>(j, i);
			}
		}
	}
	cv::Mat basU, basS, basVt;
	cv::SVD basSVD(basIntensity);

	basS = cv::Mat::zeros(3, 3, CV_32FC1);

	std::cout << basSVD.w;

	for (int i = 0; i < 3; i++)
	{
		basS.at<float>(i, i) = basSVD.w.at<float>(i, 0);
	}

	basU = basSVD.u; basVt = basSVD.vt;
//	SVDecomp(basIntensity, basU, basS, basVt);

	cv::Range rowRange, colRange;
	rowRange = cv::Range(0, imgWidth*imgHeight);
	colRange = cv::Range(0, 3);
	basU = basU(rowRange, colRange);
	rowRange = cv::Range(0, 3);
	colRange = cv::Range(0, numImagesUsed);
	basVt = basVt(rowRange, colRange);
	std::cout << "\nbasVt="<<basVt;

	cv::Mat Bstar = basU*basS;
	cv::Mat sstar = basVt;



	//GBR transform matrix
//	lambda = 1.2; 
	nu = 1.f/imgHeight; mu = 1.f/imgWidth;
	cv::Mat GBR;
	GBR = cv::Mat::zeros(3, 3, CV_32FC1);
	GBR.at<float>(0, 0) = lambda; GBR.at<float>(1, 1) = lambda; GBR.at<float>(2, 2) = 1;
	GBR.at<float>(0, 2) = -mu; GBR.at<float>(1, 2) = -nu;

	std::cout << "\nGBRMatrix=\n" << GBR;
	std::cout << "\ndeterminant(GBR) = " << determinant(GBR) << std::endl;

	cv::Mat tempB, basB;
	tempB = Bstar*GBR.inv(DECOMP_SVD);
	basB = cv::Mat(imgHeight, imgWidth, CV_32FC3);
	
	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = 0; i < imgWidth; i++)
		{
			cv::Vec3f aux;
//			aux[0] = tempB.at<float>(j*imgHeight + i,0); aux[1]= tempB.at<float>(j*imgHeight + i, 1); aux[2] = tempB.at<float>(j*imgHeight + i, 2);
			aux[2] = tempB.at<float>(j*imgHeight + i, 0); aux[0] = tempB.at<float>(j*imgHeight + i, 1); aux[1] = tempB.at<float>(j*imgHeight + i, 2);
//			aux[0] = tempB.at<float>(i*imgWidth + j, 0); aux[1] = tempB.at<float>(i*imgWidth + j, 1); aux[2] = tempB.at<float>(i*imgWidth + j, 2);
			basB.at<Vec3f>(j, i) = aux;
		}
	}
	cv::Mat basSurfaceNormal = cv::Mat(imgHeight, imgWidth, CV_32FC3);
	cv::Mat basReflectance = cv::Mat(imgHeight, imgWidth, CV_32FC1);
	cv::Mat basP_map(imgHeight, imgWidth, CV_32FC1);
	cv::Mat basQ_map(imgHeight, imgWidth, CV_32FC1);
	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = 0; i < imgWidth; i++)
		{
			float nx, ny, nz;
			float reflectance=norm(basB.at<Vec3f>(j,i));
			basReflectance.at<float>(j, i) = reflectance;
			basSurfaceNormal.at<Vec3f>(j, i) = basB.at<Vec3f>(j, i) / reflectance;
			nx = basSurfaceNormal.at<cv::Vec3f>(j, i)[0], ny = basSurfaceNormal.at<cv::Vec3f>(j, i)[1], nz = basSurfaceNormal.at<cv::Vec3f>(j, i)[2];
			basP_map.at<float>(j, i) = -(nx / nz); basQ_map.at<float>(j, i) = -(ny / nz);
		}
	}

	std::cout << "";

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//bas-Relief End
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	bool showNormal = true;
	//////////////////////////////////
	//Make Simulation Maps
	////////////////////////////////////////
	for (int k = 0; k < numData - 1; k++)
	{
		simulationMap[k] = Mat(imgWidth, imgHeight, CV_32FC1);
		for (int i = 0; i < imgHeight; i++)
		{
			for (int j = 0; j < imgWidth; j++)
			{
				cv::Vec3f normal = surfaceNormal.at<cv::Vec3f>(i, j);
				cv::Vec3f light = lightDir[k];
				float NdotL = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
				simulationMap[k].at<float>(i, j) = reflectanceFactor.at<float>(i,j)*NdotL;
			}//j
		}//i
	}//k

	cv::Mat* basSynthetic;
	basSynthetic = new cv::Mat[numData - 1];
	for (int k = 0; k < numData - 1; k++)
	{
		basSynthetic[k] = Mat(imgWidth, imgHeight, CV_32FC1);
		cv::Vec3f light = lightDir[k];
		cv::Mat lightM = cv::Mat(light);
		cv::Mat lightTrans;
		lightTrans = GBR.inv(DECOMP_SVD)*lightM;
		for (int i = 0; i < imgHeight; i++)
		{
			for (int j = 0; j < imgWidth; j++)
			{
				cv::Vec3f normal = basSurfaceNormal.at<cv::Vec3f>(i, j);
				light[0] = lightTrans.at<float>(0, 0); light[1] = lightTrans.at<float>(1, 0); light[2] = lightTrans.at<float>(2, 0);
				float NdotL = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
				basSynthetic[k].at<float>(i, j) = basReflectance.at<float>(i, j)*NdotL;
			}//j
		}//i
	}//k

	namedWindow("z-map-pFirst"); namedWindow("z-map-pFirst", CV_WINDOW_NORMAL);
	namedWindow("z-map-qFirst"); namedWindow("z-map-qFirst", CV_WINDOW_NORMAL);
	////////////////////////////
	//Reconstruct HeightMap
	///////////////////////////
	cv::Mat heightMap_xyz(imgHeight, imgWidth, CV_32FC3);
	cv::Mat heightMap_z(imgHeight, imgWidth, CV_32FC1);
	cv::Mat heightMap_zp(imgHeight, imgWidth, CV_32FC1);
	cv::Mat heightMap_zq(imgHeight, imgWidth, CV_32FC1);

	
	cv::Vec3f xyz;
	xyz[0] = 0; xyz[1] = 0;
	xyz[2] = 0;
	float ystep = 1.f / (float)imgHeight; float xstep = 1.f / (float)imgWidth;
	float qIntegral = 0;
	float pIntegral = 0;
	for (int u = 0; u < imgWidth; u++)
	{
		pIntegral += p_map.at<float>(0, u)*xstep;
		for (int v = 0; v < imgHeight; v++)
		{
			qIntegral += q_map.at<float>(v, u)*ystep;
			heightMap_zp.at<float>(v, u) = pIntegral + qIntegral;
		}
		qIntegral = 0;
	} 
	for (int v = 0; v < imgHeight; v++)
	{
		qIntegral += q_map.at<float>(v, 0)*ystep;
		for (int u = 0; u < imgHeight; u++)
		{
			pIntegral += p_map.at<float>(v, u)*xstep;
			heightMap_zq.at<float>(v, u) = pIntegral + qIntegral;
		}
		pIntegral = 0;
	} 

	heightMap_z = (heightMap_zp+heightMap_zq)/2;

	//bas z
	cv::Mat basHeightMap_xyz(imgHeight, imgWidth, CV_32FC3);
	cv::Mat basHeightMap_z(imgHeight, imgWidth, CV_32FC1);
	cv::Mat basHeightMap_zp(imgHeight, imgWidth, CV_32FC1);
	cv::Mat basHeightMap_zq(imgHeight, imgWidth, CV_32FC1);
	qIntegral = 0;
	pIntegral = 0;
	for (int u = 0; u < imgWidth; u++)
	{
		pIntegral += basP_map.at<float>(0, u)*xstep;
		for (int v = 0; v < imgHeight; v++)
		{
			qIntegral += basQ_map.at<float>(v, u)*ystep;
			basHeightMap_zp.at<float>(v, u) = pIntegral + qIntegral;
		}
		qIntegral = 0;
	}
	for (int v = 0; v < imgHeight; v++)
	{
		qIntegral += basQ_map.at<float>(v, 0)*ystep;
		for (int u = 0; u < imgHeight; u++)
		{
			pIntegral += basP_map.at<float>(v, u)*xstep;
			basHeightMap_zq.at<float>(v, u) = pIntegral + qIntegral;
		}
		pIntegral = 0;
	}

	basHeightMap_z = (basHeightMap_zp + basHeightMap_zq) / 2;

	for (int v = 0; v < imgHeight; v++)
	{
		for (int u = 0; u < imgHeight; u++)
		{
			cv::Vec3f xyz;
			xyz[0] = u*xstep; xyz[1] = v*ystep;
			xyz[2] = basHeightMap_z.at<float>(v, u);
			basHeightMap_xyz.at<Vec3f>(v, u) = xyz;
		}
	}


	for (int v = 0; v < imgHeight; v++)
	{
		for (int u = 0; u < imgHeight; u++)
		{
			cv::Vec3f xyz;
			xyz[0] = u*xstep; xyz[1] = v*ystep;
			xyz[2] = heightMap_z.at<float>(v, u);
			heightMap_xyz.at<Vec3f>(v, u) = xyz;
		}
	}

	//	heightMap_z /= -z_min;
//	heightMap_z = -heightMap_zq;
//	heightMap_z = globalHeights(p_map, q_map);
//	float min=0, max=0, min_loc, max_loc;
//	for (int u = 0; u < imgWidth; u++)
//	{
//		for (int v = 0; v < imgHeight; v++)d
//		{
//			heightMap_z.at<float>(v,u)=
//		}
//	}
//	float min, max, min_loc, max_loc;


//	for (int i = 0; i < imgHeight; i++)
//	{
//		xyz[2] = 0;
//		for (int ii = 0; ii <= i; ii++)
//		{
//			//xyz[2] += p_map.at<float>(0, ii);
//			xyz[2] += q_map.at<float>(i, 0)*ystep;
//			if (xyz[2] > max)
//				max = xyz[2];
//		}
//		xyz[0] = 0.f;
//		//xyz[1] = (float)i / (float)imgHeight;
//		xyz[1] = (float)i;
//		heightMap_xyz.at<cv::Vec3f>(i, 0) = xyz;
//		heightMap_z.at<float>(i, 0) = xyz[2];
//		for (int j = 1; j < imgWidth; j++)
//		{
////			xyz[0] = (float)j / (float)imgHeight;
//			xyz[0] = (float)j;
//			xyz[2] += p_map.at<float>(i, j)*xstep;
//			if (xyz[2] > max)
//				max = xyz[2];
//			heightMap_xyz.at<cv::Vec3f>(i, j) = xyz;
//			heightMap_z.at<float>(i, j) = xyz[2];
//			std::cout << "";
//		}
//	}
//	heightMap_z /= max;

	//Basrelief


	namedWindow("basNormal"); namedWindow("basNormal", CV_WINDOW_NORMAL);
	namedWindow("basReflec"); namedWindow("basReflec", CV_WINDOW_NORMAL);
	namedWindow("basQmap"); namedWindow("basQmap", CV_WINDOW_NORMAL);
	namedWindow("basPmap"); namedWindow("basPmap", CV_WINDOW_NORMAL);
	namedWindow("basZP"); namedWindow("basZP", CV_WINDOW_NORMAL);
	namedWindow("basZQ"); namedWindow("basZQ", CV_WINDOW_NORMAL);	namedWindow("basNormal"); namedWindow("basNormal", CV_WINDOW_NORMAL);
	namedWindow("basZ"); namedWindow("basZ", CV_WINDOW_NORMAL);

	namedWindow("synthetic"); namedWindow("synthetic", CV_WINDOW_NORMAL);


	Mat rgb[3];
	cv::split(basSurfaceNormal, rgb);


	std::cout << "\nPress ESC to proceed" << std::endl;
	while (1)
	{
		cv::Mat check;
		check = q_map - heightMap_z;
		imshow("surfaceNormal", surfaceNormal);
		imshow("faceImage", convCropImage[CVindex]);
		imshow("albedo", reflectanceFactor);
		imshow("p-map", p_map);
		imshow("q-map", q_map);
		imshow("z-map", -heightMap_z);
		imshow("z-map-pFirst", -heightMap_zp);
		imshow("z-map-qFirst", -heightMap_zq);
		imshow("simulation", simulationMap[CVindex]);
		imshow("basNormal", basSurfaceNormal);

		imshow("basReflec", basReflectance);
		imshow("basQmap", basQ_map);
		imshow("basPmap", basP_map);

		imshow("synthetic", basSynthetic[CVindex]);

		imshow("basZP", -basHeightMap_zp);
		imshow("basZQ", -basHeightMap_zq);
		imshow("basZ", -basHeightMap_z);

//		imshow("basZP", rgb[0]);
//		imshow("basZQ", rgb[1]);
//		imshow("basZ", rgb[2]);



		char ch = waitKey();
		if (ch == ',') { CVcurIndex--; CVindex = CVcurIndex % (numData - 1); }
		else if (ch == '.') { CVcurIndex++; CVindex = CVcurIndex % (numData - 1); }
		else if (ch == 27) break; //ESC

	}



#pragma endregion



	std::cout << "opencv Region End. Press Anykey to continue, Drawing on 3D" << std::endl;
	//Resource Array

	//Load Resources

	for (int i = 0; i < numData; i++)
	{
		rawData[i] = LoadImage((const char*)nDataArray[i], imageWidth[i], imageHeight[i]);
		textureData[i] = FreeImage_GetBits(rawData[i]);
	}

	//Load Resources Ends

	
	//PixelPoints to Array;

	pointCloud = new float*[imgWidth*imgHeight];
	for (int u = 0; u < imgWidth; u++)
	{
		for (int v = 0; v < imgHeight; v++)
		{
			pointCloud[u*imgHeight + v] = new float[3];
			pointCloud[u*imgHeight + v][0] = heightMap_xyz.at<Vec3f>(v, u).val[0];
			pointCloud[u*imgHeight + v][1] = heightMap_xyz.at<Vec3f>(v, u).val[1];
			pointCloud[u*imgHeight + v][2] = -heightMap_xyz.at<Vec3f>(v, u).val[2];
			
			(pc_min > pointCloud[u*imgHeight + v][2]) ? (pc_min = pointCloud[u*imgHeight + v][2]) : (pc_min);
			(pc_max < pointCloud[u*imgHeight + v][2]) ? (pc_max = pointCloud[u*imgHeight + v][2]) : (pc_max);

		}
	}
	pc_max_min = pc_max - pc_min;

	basPointCloud = new float*[imgWidth*imgHeight];
	for (int u = 0; u < imgWidth; u++)
	{
		for (int v = 0; v < imgHeight; v++)
		{
			basPointCloud[u*imgHeight + v] = new float[3];
			basPointCloud[u*imgHeight + v][0] = basHeightMap_xyz.at<Vec3f>(v, u).val[0];
			basPointCloud[u*imgHeight + v][1] = basHeightMap_xyz.at<Vec3f>(v, u).val[1];
			basPointCloud[u*imgHeight + v][2] = -basHeightMap_xyz.at<Vec3f>(v, u).val[2];

			(basPc_min > basPointCloud[u*imgHeight + v][2]) ? (basPc_min = basPointCloud[u*imgHeight + v][2]) : (basPc_min);
			(basPc_max < basPointCloud[u*imgHeight + v][2]) ? (basPc_max = basPointCloud[u*imgHeight + v][2]) : (basPc_max);

		}
	}
	basPc_max_min = basPc_max - basPc_min;

	cam = new Camera();
	initGLUT(argc, argv);

	glutMainLoop();

	for (int i = 0; i < numData; i++)
	{

	}


	destroyWindow("z-map");
	destroyWindow("faceImage");
	destroyWindow("surfaceNormal");
	destroyWindow("p-map");
	destroyWindow("q-map");
	destroyWindow("albedo");
	destroyWindow("simulation");
	destroyWindow("z-map-qFirst");
	destroyWindow("z-map-pFirst");
	destroyWindow("basNormal");
	destroyWindow("basReflec");
	destroyWindow("basQmap");
	destroyWindow("basPmap");
	destroyWindow("basZP");
	destroyWindow("basZQ");
	destroyWindow("basZ"); 
	destroyWindow("synthetic");
	//	destroyAllWindows();

	return 0;
}