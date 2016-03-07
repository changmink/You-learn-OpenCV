#define SUBJECT 1

//���� ���� ���
char	Directory[80] = "c:/cmk/miniStudy/";
char	lulu[50] = "Lulu.jpg";

#if SUBJECT == 0

#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
int main()
{
	//���� �̹���
	strcat_s(Directory, lulu);
	cv::Mat image = cv::imread(Directory);
	if (!image.data)		return 0;
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);

	// ������ �и�
	std::vector<cv::Mat> vBGR;// �迭�� ���� vector�� c++�� Ư���� �迭�̶�� ��������(���̽��� ����Ʈ����)				
	cv::split(image, vBGR);	//Mat���� 3���� ����

							// Mat �� �迭�� ��� ������
	cv::Mat mBGR[3], end;
	cv::split(image, mBGR);
	cv::merge(mBGR, 3, end);

	// �� ������ 1ä���̴�.
	cout << "Blue channel:" << vBGR[0].channels() << endl;
	cout << "Green channel:" << vBGR[1].channels() << endl;
	cout << "Red channel:" << vBGR[2].channels() << endl;

	// �и��� ������ ä���� 1�̹Ƿ� ��� ���� ������ ���ݾ� �ٸ���. �ڱ� ���κ��� �� ����(��?)
	cv::imshow("B in gray", vBGR[0]);		cv::imshow("G in gray", vBGR[1]);		cv::imshow("R in gray", vBGR[2]);
	cv::merge(vBGR, image);		cv::imshow("Restored Image", image);
	cv::waitKey();
	cv::destroyWindow("B in gray");	cv::destroyWindow("G in gray");	cv::destroyWindow("R in gray");		cv::destroyWindow("Restored Image");

	// �ٸ� ������ ���ļ� 3ä�η� �������
	cv::Mat B, G, R;
	vBGR[0].copyTo(B); vBGR[1].copyTo(G); vBGR[2].copyTo(R);// �ӽ÷� ���� �ص� ���� �ٲܰ��̱� ������

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

int main()
{
	//���� �̹���
	strcat_s(Directory, lulu);
	cv::Mat image = cv::imread(Directory);		// get source file, 
	if (!image.data) { printf("\nFail to open input file...");	return 0; }
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);
	image.convertTo(image, CV_32F, 1.0 / 255.0);//0~1������ ������ ���� �ٷ�� ������ 3��° ���� ���ϴ°�


	// ������ �и�
	std::vector<cv::Mat> vBGR;// �迭�� ���� vector�� c++�� Ư���� �迭�̶�� ��������(���̽��� ����Ʈ����)		
	cv::split(image, vBGR);		//Mat���� 3���� ���� B[0]G[1]R[2]��
	cv::Mat B, G, R;
	vBGR[0].copyTo(B);	vBGR[1].copyTo(G);	vBGR[2].copyTo(R);//opencv������ BGR�̶� RGB�� ���� ȥ�� ���� �ʵ��� ����! 

	//C,M,Y�����
	cv::Mat C, M, Y;
	R.convertTo(C, -1, -1.0, 1);//1 - K�� �ǹ� (���,Ÿ��(-1�� ����Ÿ��),���ϴ°�,���ϴ°�), 1-K���� ���������� ���� 			
	G.convertTo(M, CV_32F, -1.0, 1);
	B.convertTo(Y, CV_32F, -1.0, 1);


	cv::Mat dum0(image.rows, image.cols, CV_32FC1);
	cv::Mat dum1(image.rows, image.cols, CV_32FC1);
	cv::Mat dum2(image.rows, image.cols, CV_32FC1);
	cv::Mat vCMY[3] = { dum0, dum1, dum2 };//�Ҵ��� �ֱ����ؼ� ���

	cv::Mat Buf;
	vCMY[0] = 0;		vCMY[1] = 0;		C.copyTo(vCMY[2]);						// Make 3 channel Cyan. C=B+G
	cv::merge(vCMY, 3, Buf);		cv::imshow("Cyan", cv::Scalar(1, 1, 1) - Buf);
	vCMY[0] = 0;		M.copyTo(vCMY[1]);		vCMY[2] = 0;						// Make 3 channel Magenta. M=B+R
	cv::merge(vCMY, 3, Buf);		cv::imshow("Magenta", cv::Scalar(1, 1, 1) - Buf);
	Y.copyTo(vCMY[0]);		vCMY[1] = 0;		vCMY[2] = 0;						// Make 3 channel Yellow. Y=G+R
	cv::merge(vCMY, 3, Buf);		cv::imshow("Yellow", cv::Scalar(1, 1, 1) - Buf);
	cv::waitKey();

	//CMYK�����
	cv::Mat K, inkK;
	K = min(C, M); 	K = min(K, Y);//3�� �߿��� ���� ������
	K.convertTo(inkK, -1, -1.0, 1);	// 1 - K
	cv::namedWindow("CMYK K on white paper");	cv::imshow("CMYK K on white paper", inkK);

	//CMYK�� CMY���ϱ�
	cv::Mat C1, M1, Y1;
	C1 = C - K;				// C = 1 - R.
	M1 = M - K;				// M = 1 - G
	Y1 = Y - K;				// Y = 1 - B
	
	//push_back�� �ǵڿ� ����ִ°� ���� �Ǿ� �־ vBuf�� ũ�⸦ �˼� ���� ũ�⸦ �𸣸� merge�Ҷ� ����
	std::vector<cv::Mat> vBuf;
	vBuf.push_back(dum0);
	vBuf.push_back(dum1);
	vBuf.push_back(dum2);

	cv::Mat inkC, inkM, inkY;
	vBuf[0] = 0;		vBuf[1] = 0;		C1.copyTo(vBuf[2]);	
	cv::merge(vBuf, Buf);
	Buf.convertTo(inkC, -1, -1.0, 1);//	cv::Scalar(1, 1, 1) -  inkC�� ����
	cv::imshow("CMYK Cyan on white paper", inkC);//

	vBuf[0] = 0;		M1.copyTo(vBuf[1]);		vBuf[2] = 0;
	cv::merge(vBuf, Buf);
	Buf.convertTo(inkM, -1, -1.0, 1);
	cv::imshow("CMYK Magenta on white paper", inkM);

	Y1.copyTo(vBuf[0]);		vBuf[1] = 0;		vBuf[2] = 0;
	cv::merge(vBuf, Buf);
	Buf.convertTo(inkY, -1, -1.0, 1);
	cv::imshow("CMYK Yellow on white paper", inkY);
	cv::waitKey();   	return 0;

	// not finished ! Dispaly the combined inks.
	//cv::Mat Mix = C1 + M1 + Y1 + K;
	//cv::imshow("CMYK on white paper", Mix);

}
#endif
#if SUBJECT == 2
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	

}
#endif

#if SUBJECT == 10

//�⺻���� �ȼ� ���ٹ�
#include <iostream>
//high gui�ȿ� core�� include�Ǿ��ִ�.
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc.hpp>
using namespace std;

//at<>()�� �̿�
void changeQuarter(cv::Mat &image) {
	for (int j = 0; j < image.rows / 2; j++)
		for (int i = 0; i < image.cols / 2; i++) {
			if (image.type() == CV_8UC1) { // gray-level image
										   //(row,col)�� �����̴�. ȥ������ �ʵ��� �����Ұ�! (�̰Ͷ����� 3�ð� ���� ��)
										   //at<type>(row,col)
				image.at<uchar>(j, i) = 255;
			}
			else if (image.type() == CV_8UC3) { // color image
				//3ä�ο� ���� ���� �Ҵ���(�� �� �ּ��ϰ� ������ ����-�װ� � �ǹ��ΰ�?)
				//typedef Vec<uchar, 3> Vec3b;//uchar 3���� �̷���� ���Ͷ� ����� 
				image.at<cv::Vec3b>(j, i)[0] = 255;	//B
				image.at<cv::Vec3b>(j, i)[1] = 0;	//G
				image.at<cv::Vec3b>(j, i)[2] = 255;	//R
			}
		}
}

//ptr<>()�� �̿�
void colorReduce(cv::Mat image, int div = 64) {
	int nl = image.rows;//�� ����
	int nc = image.cols * image.channels();//�� ���� ���� �� ����

	for (int j = 0; j < nl; j++) {
		uchar* data = image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++) {
			//�÷� ����(�̰� �� ���İ�����)
			data[i] = data[i] / div*div + div / 2;//�迭 == ������ ���� ���
		}
	}
}

//���� �����ϱ� - ���� ������.(�������� ��������.)
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
	// ���� ��� �����ֱ�
	cv::Mat image = cv::imread("c:/cmk/miniStudy/Lulu.jpg");
	if (!image.data) {
		cout << "image open failed " << endl;
		exit(0);
	}
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);


	cv::Mat cq, cr, salt;
	//4���� 1�� �ٲٱ�
	image.copyTo(cq);
	changeQuarter(cq);
	cv::imshow("changeQuarter", cq);

	//�÷� ����
	image.copyTo(cr);
	colorReduce(cr);
	cv::imshow("colorReduce", cr);

	//salt Noise(�̷������� �����°� salt Noise��� �͸� �˾� �μ�)
	image.copyTo(salt);
	//cv::cvtColor(salt, salt, CV_BGR2GRAY);
	saltNoise(salt, 3000);
	cv::imshow("salt noise", salt);

	cv::waitKey(0);
}
#endif



#if SUBJECT == 11

//������ �ʿ��ѵ� , iterator�� �̿��� ����,����(�ڽ� ����)
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

	int ksize = 2 * pos + 1; //Ŀ���� ������� Ȧ��
	int KernelType = CV_32F;
	cv::Mat kernel(ksize, ksize, KernelType);

	//������ ���� ����
	for (int j = 0; j<ksize; j++)
		for (int i = 0; i<ksize; i++)
			kernel.at<float>(j, i) = 1.0 / ((float)ksize * (float)ksize);

	//������ ó���� ���� ��� iterator �̿�(const�� ������ ������ �ջ��Ű�� �ʱ� ������)
	cv::Mat_<float>::const_iterator it = kernel.begin<float>();
	cv::Mat_<float>::const_iterator itend = kernel.end<float>();


	printf("\n����ũ�� : %d X % d", ksize, ksize);

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
	//�̹��� �б�
	image = cv::imread("c:/cmk/miniStudy/Lulu.jpg");		// get source file, 
	if (!image.data)		return 0;
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);

	//Ʈ���� �߰�
	int ksize = 0;
	cv::createTrackbar("ksize", "Input Image", &ksize, 5, ChangeKernelSize);

	//�ڽ�����
	cv::Mat result;
	int ddepth = -1;
	cv::boxFilter(image, result, ddepth, cv::Size(7, 7));
	cv::imshow("boxFilter 7", result);
	cv::waitKey();

	//������ ������ �� -> ���߿� shepen ��Ű�� ��������
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


