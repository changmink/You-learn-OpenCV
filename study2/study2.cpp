#define SUBJECT 4

// ������ �̿��ϱ�
// �̹��� ����
// Ʈ���� ��� GUI

//�� ��θ� �ٲٸ� ��
char* Directory = "c:/cmk/miniStudy/"; 

#if SUBJECT == 0
//������ ����
#include <iostream>
#include "opencv2\highgui.hpp"
using namespace std;

int main() {

	//����� �̹�����
	char* FizzJPG = "Fizz.jpg";
	char* EzrealJPG = "Ezreal.jpg";
	char* LuluJPG = "Lulu.jpg";
	
	//�̹����� ��� ������ֱ�("���/����.jpg ���°� ��)
	char Fizz[100], Lulu[100], Ezreal[100];
	sprintf_s(Fizz, "%s%s", Directory, FizzJPG);
	sprintf_s(Lulu, "%s%s", Directory, LuluJPG);
	sprintf_s(Ezreal, "%s%s", Directory, EzrealJPG);
	cout << Fizz << endl;

	// �׷��� �������� ����
	cv::Mat image = cv::imread(Fizz, CV_LOAD_IMAGE_GRAYSCALE);
	cv::namedWindow("original in gray");
	cv::imshow("original in gray", image);
	cv::waitKey(0);


	// ���� ���� �����ϱ� - �̰� �� �Ǵ°�?
	cv::Mat result;

	result = image + 50; //�� ���ϸ� �������?
	cv::imshow("image, +", result); cv::waitKey(0);

	result = image - 50;
	cv::imshow("image, -", result); cv::waitKey(0);

	result = image * 1.3;
	cv::imshow("image, multiplied by greater than 1", result); cv::waitKey(0);

	cv::Mat result1 = image * 0.7;
	cv::imshow("result 1 - image, multiplied by less than 1", result1); cv::waitKey(0);

	image = image + 250;			// ���� ������ ��ȭ���ѵ� result ������ �ٲ��� �ʴ´�. �ܼ� assign���� ���̰� �ִ�.
	cv::imshow("result 1 - again", result1); cv::waitKey(0);


	// read in color
	cv::Mat imageA = cv::imread(Lulu, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow("imageA in color"); cv::imshow("imageA in color", imageA); cv::waitKey(0);

	cv::Mat result2;
	int b = 0; int g = 100; int r = 0;
	result2 = imageA + cv::Scalar(b, g, r);
	cv::imshow("G channel of imageA is added", result2); cv::waitKey(0);

	b = 0; g = 0; r = 100;
	result = imageA + cv::Scalar(b, g, r);
	cv::imshow("R channel of imageA is added", result); cv::waitKey(0);

	cv::Mat imageB = cv::imread(Ezreal, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow("imageB in color"); cv::imshow("imageB in color", imageB); cv::waitKey(0);

	// �� ������ �Ϸ��� �̹����� ũ�Ⱑ ���ƾ� �Ѵ� ��? - ����� ������ �����غ�
	cv::Mat imageC = (imageA + imageB)/2; 	cv::imshow("(A+B)/2", imageC); cv::waitKey(0);

	return 0;
}

#endif

#if SUBJECT == 1
#include <opencv2\highgui.hpp>//core.hpp�� ���Ե�
#include <iostream>
using namespace std;

int main(void)
{
	char OutputVideo[100];
	sprintf_s(OutputVideo,"%s%s",Directory,"Output.avi");
	
	cv::VideoCapture cap; //���� ĸó�� ����, ī�޶� ���� ���

	cap.open(0);	//ī�޶� 0�̸� ���� 0	
	
	if (!cap.isOpened()) {				
		printf("\nFail to open camera");
		return -1;
	}
	
	//������ ���� ���

	double fps;
	fps = cap.get(CV_CAP_PROP_FPS);
	cout << "\nFPS = " << fps << endl; //ī�޶�� 0

	cv::Size FrameSize;
	
	FrameSize.width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	FrameSize.height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	cout << "\nWidth * Height = " << FrameSize.width << "*" << FrameSize.height << endl;
	

	//���� ������ ����
	cv::VideoWriter writer; 
	fps = 30;
	writer.open(OutputVideo, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps ,FrameSize);
	//cv::VideoWriter::fourcc('D','I','V','X')�� ���� �ɹ� �Լ� http://www.soen.kr/lecture/ccpp/cpp3/27-3-3.htm

	//�ϳ��� ���
	cv::Mat frame;//������ �����ְ� �Ϸ��� �ᱹ Mat�� �ʿ�
	cap.read(frame);
	imshow("One shot", frame);

	//�������� ���
	for (;;)
	{
		cap.read(frame);
		imshow("Camera Video", frame);
		writer.write(frame);
		if (cv::waitKey(1) == 0x1b) break; //ESC
	}

	return 0;
}
#endif

#if SUBJECT == 2
#include <opencv2\highgui.hpp>
#include <iostream>
using namespace std;
int main() {
	char*  movie= "the_return_of_the_king.avi";
	char Video[100];
	sprintf_s(Video, "%s%s", Directory, movie);
	cv::VideoCapture cap;
	cap.open(Video);

	if (!cap.isOpened()) {
		printf("\nFail to open camera");
		return -1;
	}

	double fps = cap.get(CV_CAP_PROP_FPS);
	
	cout << fps << ","<<cap.get(CV_CAP_PROP_FRAME_COUNT) <<endl;
	
	cv::Size FrameSize;

	FrameSize.width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	FrameSize.height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	cout << "\nWidth * Height = " << FrameSize.width << "*" << FrameSize.height << endl;

	cv::Mat frame;
	while (1) { 
		cap.read(frame);
		cv::imshow("Video",frame);
		if (cv::waitKey(1) == 0x1b) break;
		//if (cv::)break;
	}

}
#endif

#if SUBJECT == 3
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Global variables
Mat img;									

void onTrackbarSlide(int pos, void *)
{
	Mat dst;

	threshold(img, dst, pos, 255, THRESH_BINARY);
	imshow("Thresolded Image", dst);

	threshold(img, dst, pos, 128, THRESH_TRUNC);
	imshow("Truncated Image", dst);

	if (pos > 128 && img.channels() == 3)
		cvtColor(img, dst, CV_BGR2GRAY);
	else dst = img;
	imshow("Gray or Color", dst);

}

int main()
{
	char* LuluJPG = "Lulu.jpg";
	char Lulu[100];
	sprintf_s(Lulu, "%s%s", Directory, LuluJPG);

	img = imread(Lulu);	//�⺻ 8UC3

	if (img.data == NULL) { printf("\nNo image found!\n"); return(-1); }

	namedWindow("Trackbar Example"); 	imshow("Trackbar Example", img);

	int		slider_max = 255;	//�ִ�
	int		slider_position = 0; // ����

	//����! �����̴� ��ġ�� ���۷���(&)�� �������Ѵ�!! -> �����̴��� �����϶����� �ٲ�ϱ�
	createTrackbar("Threshold", "Trackbar Example", &slider_position, slider_max, onTrackbarSlide);
	
	while (char(waitKey(1)) != 0x1b) { // esc�� ������  
		cout << slider_position << endl;
	}
	return 0;
}

#endif

#if SUBJECT == 4
#include "opencv2/highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <iostream>
using namespace std;

//SAD(Sum of Absolute Differences)�� �̿��ؼ� ���̸� �˾Ƴ���

int main(void)
{
	
	cv:: VideoCapture cap;

	cap.open(0);
	
	if (!cap.isOpened()) {				// Check if the camera is opened sucessfully.
		printf("\nFail to open camera");
		return -1;
	}

	double fps = cap.get(CV_CAP_PROP_FPS); 	printf("\nFPS = %f", fps);

	cv::Size FrameSize;

	FrameSize.width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	FrameSize.height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf("\nWidth * Height = %d * %d", FrameSize.width, FrameSize.height);

	fps = 60;

	cv::Mat frame, frameCurrent, dst;
	cv::Mat framePrev(FrameSize.height, FrameSize.width, CV_8UC1);

	cv::namedWindow("original", 1);

	for (;;)
	{
		cap.read(frame);
		imshow("original", frame);
		
		cv::cvtColor(frame, frameCurrent, CV_BGR2GRAY);//������� �ٲ�
		
		cv::absdiff(frameCurrent, framePrev, dst);//�� ������ ���̸� ����(���밪)
		cv::Scalar SAD = sum(dst) / (FrameSize.width * FrameSize.height);//SAD�� ���̰� Ŭ���� Ŀ����.
		//sum() - ������ �ȼ� ���� ��� ��ģ��.

		cout << SAD << endl;

		float	threshold = 8;

		if (SAD.val[0] > threshold)	//�Ӱ谪���� ũ�� ����
			printf("\nSAD = %f : Motion detected", SAD.val[0]);		// val[0] means the first element of Scalar.

		frameCurrent.copyTo(framePrev);	 // Store current frame to previous one.

		if (cv::waitKey(1000 / fps) == 0x1b) break;			// Break if key input is escape code.

	}
	return 0;
}

#endif

#if SUBJECT == 5
#include "opencv2/highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <iostream>
using namespace std;

//Mat���� ����

int main(void)
{
	cv::Mat original;
	cv::Mat copy1, copy2;

	char* LuluJPG = "Lulu.jpg";
	char Lulu[100];
	sprintf_s(Lulu, "%s%s", Directory, LuluJPG);

	//���� �ϱ�
	original = cv::imread(Lulu);
	copy1 = original;
	original.copyTo(copy2);

	cv::imshow("original", original);
	cv::imshow("copy1", copy1);
	cv::imshow("copy2", copy2);
	cv::waitKey(0);

	original = original + 100;
	cv::imshow("original", original);
	cv::imshow("copy1", copy1);
	cv::imshow("copy2", copy2);
	cv::waitKey(0);
	return 0;
}

#endif