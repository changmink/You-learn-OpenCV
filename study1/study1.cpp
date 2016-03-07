#define STEP 3

//자신의 경로에 맞게 설정하세요
char *fileName = "c:/cmk/miniStudy/Lulu.jpg";

#if STEP == 0
//가장 기본적인 프로그램

#include "opencv2\highgui.hpp"
//#include "opencv2\core.hpp"//highgui 안에 core.hpp가 인클루드되어 있음
#include <iostream>
using namespace std;

int main(){
	//이미지 열기
	cv::Mat image = cv::imread(fileName);

	//에러 체크
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}
	//이미지 보여주기
	cv::namedWindow("Hello OpenCV");
	cv::imshow("Hello OpenCV!", image);
	cv::waitKey(0);
}
#endif

#if STEP == 1
//이미지 열 때 플래그, Mat형의 속성들
//mat참조: https://sites.google.com/site/opencvwikiproject/table-of-contents/opencv-api-reference/core-the-core-functionality/basic-structures/mat
#include "opencv2\highgui.hpp"
#include <iostream>
using namespace std;

int main(){

	//int	flag = CV_LOAD_IMAGE_UNCHANGED;		// =-1. 8bit, color or gray. 파일 칼라(혹은 모노) 그대로.
	//int	flag = CV_LOAD_IMAGE_GRAYSCALE;			// =0. 8bit, gray 
	int	flag = CV_LOAD_IMAGE_COLOR;			// =1. ?, color. 모노 영상이라도 3채널 칼라 영상으로 받아들임.
	//int	flag = CV_LOAD_IMAGE_ANYDEPTH;		// =2. any depth, ?. 파일이 가진 depth 그대로. 
	//int	flag = CV_LOAD_IMAGE_ANYCOLOR;		// =4. ?, any color
	cout << "flag:" << flag << endl;

	//이미지 열기
	cv::Mat image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

	//이미지 속성
	cout << "행: " << image.rows << " 열: " << image.cols << endl;
	cout << "가로: " << image.size().width << " 세로: " << image.size().height << endl;
	cout << "크기: " << image.size() << endl;
	cout << "채널: " << image.channels() << endl;
	cout << "뎁스: " << image.depth() << ", "; //CV_8U가 0으로 define되어 있어서 0으로 나옴
	if (image.depth() == CV_8U)
		cout << "8bit unsigned char입니다." << endl;
	else if (image.depth() == CV_16U)
		cout << "16bit unsigned char입니다." << endl;
	else if (image.depth() == CV_32F)
		cout << "16bit unsigned float입니다." << endl;

	//에러 체크
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}

	//이미지 보여주기
	cv::namedWindow("first");
	cv::imshow("first", image);
	cv::waitKey(0);//0은 무한정 기다림 ms단위만 큼 키 대기
}
#endif

#if STEP == 2
#include "opencv2\highgui.hpp"
#include <iostream>
using namespace std;
int main(){

	//이미지 열기
	cv::Mat image = cv::imread(fileName);

	//에러 체크
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}

	//이미지 뒤집기
	cv::Mat fliped_0, fliped_pos, fliped_neg;
	cv::flip(image, fliped_0, 0);//좌우 대칭
	cv::flip(image, fliped_pos, 1);//상하 대칭(양수)
	cv::flip(image, fliped_neg, -1);//둘다 대칭(음수)

	//이미지 보여주기(namedWindow()없이도 imshow할수 있다!)
	cv::imshow("original", image);
	if (cv::waitKey(0) == 's')
		cv::imwrite("ez.jpg", fliped_pos);
	cv::imshow("fliped_0", fliped_0);
	cv::waitKey(0);
	cv::imshow("fliped_pos", fliped_pos);
	cv::waitKey(0);
	cv::imshow("fliped_neg", fliped_neg);

	cv::waitKey(0);
}
#endif

#if STEP == 3

//마우스 콜백(클릭한 부분의 색상 정보를 알려준다.)과 이미지에 그림그리기, 영상 생성하기
//c++ 캐스트 연산자들 http://prostars.net/65

#include "opencv2\highgui.hpp"
//circle,puttext등을 사용하기 위한 헤더 추가
#include "opencv2\imgproc.hpp"
#include <iostream>
using namespace std;

void onMouse(int event, int x, int y, int flags, void *param){
	/*마우스 콜백함수: 클릭한 부분의 색정보를 알려줌*/
	cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
	switch (event){
	case CV_EVENT_LBUTTONDOWN:
		std::cout << "at (" << x << "," << y << ") value is "
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << endl;
		break;
	}
}

cv::Mat makeImageBGR(int b = 0, int g = 0, int r = 0) {
	/* BGR 값을 입력받아 해당하는 영상을 만든다.*/
	cv::Mat ima(300, 240, CV_8UC3, cv::Scalar(b, g, r));
	return ima;
}

int main(){

	//이미지 열기
	cv::Mat image = cv::imread(fileName);

	//에러 체크
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}

	//이미지에 원 그리기 
	cv::circle(image,		// 목적영상
		cv::Point(155, 110),// 중심
		65,					//반지름
		255,				//색
		3					//두께
		);
	cv::putText(image, "Target", cv::Point(100, 200), cv::FONT_HERSHEY_PLAIN, 2.0, 255, 2);

	//마우스 콜백 달기
	cv::namedWindow("target");
	cv::setMouseCallback("target", onMouse, reinterpret_cast<void*>(&image));
	

	//이미지 보여주기
	cv::imshow("target", image);
	cv::waitKey(0);

	//영상 생성&보여주기(위 영상과의 차이?)
	cv::Mat maden = makeImageBGR(128, 255, 0);
	cv::imshow("maden",maden);
	cv::waitKey(0);
}
#endif

#if STEP == 4
/*
문제
1. 주어진 2개의 영상을 열어라
2. 원을 캐릭터 얼굴에 맞추어라
3. 열린영상과 같은 크기의 영상을 만들어라
4. 인터넷에서 사각형을 그리는 함수를 찾아서 영상에 사각형을 그려라
5. 좌우로 뒤집혀진 모노(흑백) 영상을 만들어라
*/
#endif