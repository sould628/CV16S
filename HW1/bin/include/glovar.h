#pragma once

#define mPI 3.14159265358979323846

//must be equal or larger than 3
int numImagesUsed = 11;




//opencv
cv::Mat *image, *cropImage;
int imgWidth, imgHeight;

//opengl
float **pointCloud;	float pc_min = 0, pc_max = 0, pc_max_min;
float **basPointCloud;	float basPc_min = 0, basPc_max = 0, basPc_max_min;

Camera *cam; GLint button_id; GLfloat click_pos[2];

int curIndex = 0, index; int renderMode=2;

int *imageWidth, *imageHeight;
int windowWidth = 800, windowHeight = 800;


GLubyte** textureData;
FIBITMAP** rawData;

//GBR parameters
float lambda=2, mu=0.003, nu=0.003;


extern const char* frontal_face_classifier_path = "./resource/haarcascade_frontalface_default.xml";

extern const int numData = 12;

extern char* nData0 = "./resource/yaleB01_P00_Ambient.pgm";
extern char* nData1 = "./resource/yaleB01_P00A+005E+10.pgm";
extern char* nData2 = "./resource/yaleB01_P00A+005E-10.pgm";
extern char* nData3 = "./resource/yaleB01_P00A+020E-40.pgm";
extern char* nData4 = "./resource/yaleB01_P00A+035E+15.pgm";
extern char* nData5 = "./resource/yaleB01_P00A+035E+40.pgm";
extern char* nData6 = "./resource/yaleB01_P00A+060E+20.pgm";
extern char* nData7 = "./resource/yaleB01_P00A-005E+10.pgm";
extern char* nData8 = "./resource/yaleB01_P00A-005E-10.pgm";
extern char* nData9 = "./resource/yaleB01_P00A-010E-20.pgm";
extern char* nData10 = "./resource/yaleB01_P00A-020E-40.pgm";
extern char* nData11 = "./resource/yaleB01_P00A-035E+40.pgm";

extern float azimuth[12] = { 0.0f, 5.f, 5.f, 20.f, 35.f, 35.f, 60.f, -5.f, -5.f, -10.f, -20.f, -35.f };
extern float elevation[12]{ 0.0f, 10.f, -10.f, -40.f, 15.f, 40.f, 20.f, 10.f, -10.f, -20.f, -40.f, 40.f };