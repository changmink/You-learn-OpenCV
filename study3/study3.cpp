#define SUBJECT 0

//사용될 파일 경로
char	Directory[80] = "D:\\GoogleDrive\\miniStudy\\images/";
char	lulu[50] = "Lulu.jpg";	

#if SUBJECT == 0

#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
int main()
{
	//원본 이미지
	strcat_s(Directory, lulu);
	cv::Mat image = cv::imread(Directory);
	if (!image.data)		return 0;
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);

	// 영상을 분리
	std::vector<cv::Mat> vBGR;// 배열을 만듬 vector는 c++의 특별한 배열이라고 이해하자(파이썬의 리스트같은)				
	cv::split(image, vBGR);	//Mat형이 3개가 나옴

	// Mat 형 배열도 사용 가능함
	cv::Mat mBGR[3], end;
	cv::split(image, mBGR);
	cv::merge(mBGR, 3, end);

	// 각 영상은 1채널이다.
	cout << "Blue channel:" << vBGR[0].channels() << endl;
	cout << "Green channel:" << vBGR[1].channels() << endl;
	cout << "Red channel:" << vBGR[2].channels() << endl;

	// 분리된 영상은 채널이 1이므로 흑백 영상 하지만 조금씩 다르다. 자기 색부분이 더 밝음(왜?)
	cv::imshow("B in gray", vBGR[0]);		cv::imshow("G in gray", vBGR[1]);		cv::imshow("R in gray", vBGR[2]);
	cv::merge(vBGR, image);		cv::imshow("Restored Image", image);
	cv::waitKey();
	cv::destroyWindow("B in gray");	cv::destroyWindow("G in gray");	cv::destroyWindow("R in gray");		cv::destroyWindow("Restored Image");

	// 다른 영상을 합쳐서 3채널로 만들어줌
	cv::Mat B, G, R;
	vBGR[0].copyTo(B); vBGR[1].copyTo(G); vBGR[2].copyTo(R);// 임시로 복사 해둠 값을 바꿀것이기 때문에

	B.copyTo(vBGR[0]);		vBGR[1] = 0;		vBGR[2] = 0;
	cv::merge(vBGR, image);			cv::imshow("Blue", image);

	vBGR[0] = 0;		G.copyTo(vBGR[1]);		vBGR[2] = 0;
	cv::merge(vBGR, image);			cv::imshow("Green", image);

	vBGR[0] = 0;		vBGR[1] = 0;			R.copyTo(vBGR[2]);
	cv::merge(vBGR, image);			cv::imshow("Red", image);

	cv::waitKey();
	cv::destroyAllWindows();

	return 0;
}


#endif

#if SUBJECT == 1
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
int main()
{
	//원본 이미지
	strcat_s(Directory, lulu);
	cv::Mat image = cv::imread(Directory);		// get source file, 
	if (!image.data) { printf("\nFail to open input file...");	return 0; }
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);
	image.convertTo(image, CV_32F, 1.0 / 255.0);//0~1사이의 값으로 만들어서 다루기 쉽게함 3번째 값은 곱하는것


	// 영상을 분리
	std::vector<cv::Mat> vBGR;// 배열을 만듬 vector는 c++의 특별한 배열이라고 이해하자(파이썬의 리스트같은)		
	cv::split(image, vBGR);		//Mat형이 3개가 나옴 B[0]G[1]R[2]순
	cv::Mat B, G, R;
	vBGR[0].copyTo(B);	vBGR[1].copyTo(G);	vBGR[2].copyTo(R);//opencv에서는 BGR이라서 RGB랑 순서 혼동 하지 않도록 주의! 

	//C,M,Y만들기
	cv::Mat C, M, Y;
	//1 - K의 의미 (결과,타입(-1은 같은타입),곱하는값,더하는값), 1-K보다 안정적으로 수행 
	R.convertTo(C, -1, -1.0, 1);// C = 1-R			
	G.convertTo(M, CV_32F, -1.0, 1);
	B.convertTo(Y, CV_32F, -1.0, 1);

	//할당해 주기위해서 사용
	cv::Mat dum0(image.rows, image.cols, CV_32FC1);
	cv::Mat dum1(image.rows, image.cols, CV_32FC1);
	cv::Mat dum2(image.rows, image.cols, CV_32FC1);
	cv::Mat vCMY[3] = { dum0, dum1, dum2 };

	//CMY출력 CMY의 원래색을 보려면 1에서 빼주어야함(BGR영역에서 출력하므로)
	cv::Mat Buf;
	vCMY[0] = 0;		vCMY[1] = 0;		C.copyTo(vCMY[2]);						
	cv::merge(vCMY, 3, Buf);		cv::imshow("Cyan", cv::Scalar(1, 1, 1) - Buf);
	vCMY[0] = 0;		M.copyTo(vCMY[1]);		vCMY[2] = 0;				
	cv::merge(vCMY, 3, Buf);		cv::imshow("Magenta", cv::Scalar(1, 1, 1) - Buf);
	Y.copyTo(vCMY[0]);		vCMY[1] = 0;		vCMY[2] = 0;
	cv::merge(vCMY, 3, Buf);		cv::imshow("Yellow", cv::Scalar(1, 1, 1) - Buf);
	cv::waitKey();

	//CMYK만들기
	cv::Mat K, inkK;
	K = min(C, M); 	K = min(K, Y);//3개 중에서 제일 작은거
	K.convertTo(inkK, -1, -1.0, 1);	// 1 - K = RGB영역에서의 K의 색
	cv::namedWindow("CMYK K on white paper");	cv::imshow("CMYK K on white paper", inkK);

	//CMYK의 CMY구하기
	cv::Mat C1, M1, Y1;
	C1 = C - K;				// C = 1 - R.
	M1 = M - K;				// M = 1 - G
	Y1 = Y - K;				// Y = 1 - B
	
	//push_back은 맨뒤에 집어넣는것 선언만 되어 있어서 vBuf의 크기를 알수 없음 크기를 모르면 merge할때 오류
	std::vector<cv::Mat> vBuf;
	vBuf.push_back(dum0);
	vBuf.push_back(dum1);
	vBuf.push_back(dum2);

	cv::Mat inkC, inkM, inkY;
	vBuf[0] = 0;		vBuf[1] = 0;		C1.copyTo(vBuf[2]);	
	cv::merge(vBuf, Buf);
	Buf.convertTo(inkC, -1, -1.0, 1);//	cv::Scalar(1, 1, 1) -  inkC와 같음
	cv::imshow("CMYK Cyan", inkC);//

	vBuf[0] = 0;		M1.copyTo(vBuf[1]);		vBuf[2] = 0;
	cv::merge(vBuf, Buf);
	Buf.convertTo(inkM, -1, -1.0, 1);
	cv::imshow("CMYK Magenta", inkM);

	Y1.copyTo(vBuf[0]);		vBuf[1] = 0;		vBuf[2] = 0;
	cv::merge(vBuf, Buf);
	Buf.convertTo(inkY, -1, -1.0, 1);
	cv::imshow("CMYK Yellow", inkY);
	cv::waitKey(0);

	//CMYK 결합
	cv::Mat Mix = C1 + M1 + Y1 + K;// 잘못됨

	Y1.copyTo(vBuf[0]);	M1.copyTo(vBuf[1]);	C1.copyTo(vBuf[2]);
	vBuf[0] += K; vBuf[1] += K;	vBuf[2] += K;
	cv::merge(vBuf, Mix);
	
	cv::imshow("CMYK", cv::Scalar(1, 1, 1) - Mix);
	cv::waitKey(0);   	return 0;

}
#endif
#if SUBJECT == 2
//HSV
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	//원본 이미지
	strcat_s(Directory, lulu);
	cv::Mat image = cv::imread(Directory);		// get source file, 
	if (!image.data) { printf("\nFail to open input file...");	return 0; }
	cv::imshow("1. Input Image", image);
	image.convertTo(image, CV_32F, 1.0 / 255.0);						// scale image as 0~1


	cv::Mat dst, Strong, Weak;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 2.  Strengthen the Color by increasing S in HSV.
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	float	RatioS = 1.5;									// decreasing or increasing ratio
	cv::cvtColor(image, dst, CV_BGR2HSV);//간단하게 HSV로 바꿀수 있다.
	cv::Mat vHSV[3];
	cv::split(dst, vHSV);										// Seperate to Mat array
	vHSV[1] *= RatioS;												// empahasize the saturation
	cv::merge(vHSV, 3, dst);									// Merge 3 planes to a Mat.
	cv::cvtColor(dst, Strong, CV_HSV2BGR);//우리가
	cv::imshow("2. Strengthened Color", Strong);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3.  Weaken the Color by decreasing S in HSV.
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::cvtColor(image, dst, CV_BGR2HSV);				// Convert to HSV
	cv::split(dst, vHSV);										// Seperate to Mat array
	vHSV[1] *= 1.0 / RatioS;								// dim the saturation
	cv::merge(vHSV, 3, dst);									// Merge 3 planes to a Mat.
	cv::cvtColor(dst, Weak, CV_HSV2BGR);				// Convert HSV to BGR
	cv::imshow("3. Weakened Color", Weak);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 4.  Restore the Color by increasing S in HSV.  
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	vHSV[1] *= RatioS;									// empahasize the saturation
	cv::merge(vHSV, 3, dst);									// Merge 2 planes to a Mat.
	cv::cvtColor(dst, Strong, CV_HSV2BGR);				// Convert HSV to BGR
	cv::imshow("4. Restored from weakened color", Strong);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 5.  Increase brightness by increasing V in HSV.
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	float RatioVup = 2;
	float RatioVdown = 6;
	cv::cvtColor(image, dst, CV_BGR2HSV);				// Convert to HSV, again!
	cv::split(dst, vHSV);										// Seperate Mat array
	vHSV[2] *= RatioVup;									// increase V
	cv::merge(vHSV, 3, dst);									// Merge 3 planes to a Mat.
	cv::cvtColor(dst, dst, CV_HSV2BGR);					// Convert HSV to BGR
	cv::imshow("5. Brighter Image", dst);

	vHSV[2] *= 1.0 / (RatioVdown);								// decrease V
	cv::merge(vHSV, 3, dst);									// Merge 3 planes to a Mat.
	cv::cvtColor(dst, dst, CV_HSV2BGR);					// Convert HSV to BGR
	cv::imshow("5. Darker Image", dst);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 6.  Increase both S and V together
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::cvtColor(image, dst, CV_BGR2HSV);				// Convert to HSV, again!
	cv::split(dst, vHSV);										// Seperate Mat array
	vHSV[1] *= RatioS;										// increase S
	vHSV[2] *= RatioVup;									// increase V
	cv::merge(vHSV, 3, dst);									// Merge 3 planes to a Mat.
	cv::cvtColor(dst, dst, CV_HSV2BGR);					// Convert HSV to BGR
	cv::imshow("6. Brighter and Color emphasized Image", dst);


	cv::waitKey();   	return 0;
}
#endif

#if SUBJECT == 3
//HSV
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	// Read input image
	strcat_s(Directory, lulu);
	cv::Mat image = cv::imread(Directory);		// get source file, 
	if (!image.data) { printf("\nFail to open input file...");	return 0; }
	cv::imshow("Input Image", image);
	image.convertTo(image, CV_32F, 1.0 / 255.0);

	cv::Mat dst, dst2;

	//image *= 1./255;
	cvtColor(image, dst, CV_BGR2HLS);

	// Split the image
	std::vector<cv::Mat> imageHLS, HLS;
	cv::split(dst, imageHLS);			// imageHLS[0], imageHLS[1], imageHLS[2]

	HLS = imageHLS;					// !!!!! They point to the same data content. This assignment does not copy data.
	HLS[0] += 180 / 4;					// Rotate hue 90 degrees in counterclockwise. When you are using CV_8U data, value 180 is 360 degrees.
	cv::merge(HLS, dst2);
	cvtColor(dst2, dst2, CV_HLS2BGR);
	cv::namedWindow("Hue rotated by 90 degrees"); cv::imshow("Hue rotated by 90 degrees", dst2);

	HLS[0] += 180 / 4;					// Rotate hue 90 degrees again. total 180.
	cv::merge(HLS, dst2);
	cvtColor(dst2, dst2, CV_HLS2BGR);
	cv::namedWindow("Hue rotated by 180 degrees"); cv::imshow("Hue rotated by 180 degrees", dst2);

	HLS[0] += 180 / 4;					// Rotate hue 90 degrees again. total 270. 
	cv::merge(HLS, dst2);
	cvtColor(dst2, dst2, CV_HLS2BGR);
	cv::namedWindow("Hue rotated by 270 degrees"); cv::imshow("Hue rotated by 270 degrees", dst2);

	HLS[0] += 180 / 4;					// Rotate hue 90 degrees again. total 360.
	cv::merge(HLS, dst2);
	cvtColor(dst2, dst2, CV_HLS2BGR);
	cv::namedWindow("Hue rotated by 360 degrees"); cv::imshow("Hue rotated by 360 degrees", dst2);
	cv::waitKey();
	return 0;
}
#endif
#if SUBJECT == 4

//기본적인 픽셀 접근법
#include <iostream>
//high gui안에 core가 include되어있다.
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc.hpp>
using namespace std;

//at<>()을 이용
void changeQuarter(cv::Mat &image) {
	for (int j = 0; j < image.rows / 2; j++)
		for (int i = 0; i < image.cols / 2; i++) {
			if (image.type() == CV_8UC1) { // gray-level image
										   //(row,col)의 순서이다. 혼동하지 않도록 조심할것! (이것때문에 3시간 버림 ㅠ)
										   //at<type>(row,col)
				image.at<uchar>(j, i) = 255;
			}
			else if (image.type() == CV_8UC3) { // color image
				//3채널에 각각 값을 할당함(한 줄 주석하고 실행해 보기-그게 어떤 의미인가?)
				//typedef Vec<uchar, 3> Vec3b;//uchar 3개로 이루어진 백터라 보면됨 
				image.at<cv::Vec3b>(j, i)[0] = 255;	//B
				image.at<cv::Vec3b>(j, i)[1] = 0;	//G
				image.at<cv::Vec3b>(j, i)[2] = 255;	//R
			}
		}
}

//ptr<>()을 이용
void colorReduce(cv::Mat image, int div = 64) {
	int nl = image.rows;//행 개수
	int nc = image.cols * image.channels();//각 행의 원소 총 개수

	for (int j = 0; j < nl; j++) {
		uchar* data = image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++) {
			//컬러 감소(이건 걍 공식같은거)
			data[i] = data[i] / div*div + div / 2;//배열 == 포인터 임을 기억
		}
	}
}

//직접 접근하기 - 제일 빠르다.(안정성이 떨어진다.)
void saltNoise(cv::Mat image, int n) {

	int i, j;
	for (int k = 0; k<n; k++) {
		i = std::rand() % image.cols;
		j = std::rand() % image.rows;

		if (image.type() == CV_8UC1) { // gray-level image
			uchar* data = (uchar*)image.data;
			//data[row* image.cols + col]
			data[j * image.cols + i] = 255;
		}
		else if (image.type() == CV_8UC3) { // color image
			cv::Vec3b* data = (cv::Vec3b*)image.data;
			data[j * image.cols + i][0] = 255;
			data[j * image.cols + i][1] = 255;
			data[j * image.cols + i][2] = 255;
		}
	}
}

int main() {
	// 파일 열어서 보여주기
	cv::Mat image = cv::imread("c:/cmk/miniStudy/Lulu.jpg");
	if (!image.data) {
		cout << "image open failed " << endl;
		exit(0);
	}
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);


	cv::Mat cq, cr, salt;
	//4분의 1만 바꾸기
	image.copyTo(cq);
	changeQuarter(cq);
	cv::imshow("changeQuarter", cq);

	//컬러 감소
	image.copyTo(cr);
	colorReduce(cr);
	cv::imshow("colorReduce", cr);

	//salt Noise(이런식으로 나오는게 salt Noise라는 것만 알아 두셈)
	image.copyTo(salt);
	//cv::cvtColor(salt, salt, CV_BGR2GRAY);
	saltNoise(salt, 3000);
	cv::imshow("salt noise", salt);

	cv::waitKey(0);
}
#endif



#if SUBJECT == 5

//정리가 필요한듯 , iterator를 이용한 접근,필터(박스 필터)
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


void saltNoise(cv::Mat image, int n) {

	int i, j;
	for (int k = 0; k<n; k++) {
		i = std::rand() % image.cols;
		j = std::rand() % image.rows;

		if (image.type() == CV_8UC1) { // gray-level image
			uchar* data = (uchar*)image.data;
			data[j * image.cols + i] = 255;
		}
		else if (image.type() == CV_8UC3) { // color image
			cv::Vec3b* data = (cv::Vec3b*)image.data;
			data[j * image.cols + i][0] = 255;
			data[j * image.cols + i][1] = 255;
			data[j * image.cols + i][2] = 255;
		}
	}
}
cv::Mat image;
void ChangeKernelSize(int pos, void*) {

	int ksize = 2 * pos + 1; //커널의 사이즈는 홀수
	int KernelType = CV_32F;
	cv::Mat kernel(ksize, ksize, KernelType);

	//필터의 값을 설정
	for (int j = 0; j<ksize; j++)
		for (int i = 0; i<ksize; i++)
			kernel.at<float>(j, i) = 1.0 / ((float)ksize * (float)ksize);

	//필터의 처음과 끝을 얻기 iterator 이용(const인 이유는 원본을 손상시키지 않기 위함임)
	cv::Mat_<float>::const_iterator it = kernel.begin<float>();
	cv::Mat_<float>::const_iterator itend = kernel.end<float>();


	printf("\n필터크기 : %d X % d", ksize, ksize);

	for (int i = 0; it != itend; ++it) {
		if (i++ % ksize == 0) 		std::cout << std::endl;
		printf("  %4.2f", *it);
	}

	int ddepth = -1;	// the output image will have the same depth as the source.
	cv::Mat dst;		// output array

	filter2D(image, dst, ddepth, kernel);				//, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT )
	cv::namedWindow("Filter2D result"); 	cv::imshow("Filter2D result", dst);
}
int main()
{
	//이미지 읽기
	image = cv::imread("c:/cmk/miniStudy/Lulu.jpg");		// get source file, 
	if (!image.data)		return 0;
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);

	//트랙바 추가
	int ksize = 0;
	cv::createTrackbar("ksize", "Input Image", &ksize, 5, ChangeKernelSize);

	//박스필터
	cv::Mat result;
	int ddepth = -1;
	cv::boxFilter(image, result, ddepth, cv::Size(7, 7));
	cv::imshow("boxFilter 7", result);
	cv::waitKey();

	//노이즈 제거의 예 -> 나중에 shepen 시키면 선명해짐
	cv::Mat saltN;
	image.copyTo(saltN);
	saltNoise(saltN, 1000);
	cv::imshow("Noise", saltN);
	cv::boxFilter(saltN, saltN, ddepth, cv::Size(7, 7));
	cv::imshow("Noise after blur", saltN);
	cv::waitKey();

	return 0;
}
#endif


