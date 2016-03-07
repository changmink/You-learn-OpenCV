#define STEP 3

//�ڽ��� ��ο� �°� �����ϼ���
char *fileName = "c:/cmk/miniStudy/Lulu.jpg";

#if STEP == 0
//���� �⺻���� ���α׷�

#include "opencv2\highgui.hpp"
//#include "opencv2\core.hpp"//highgui �ȿ� core.hpp�� ��Ŭ���Ǿ� ����
#include <iostream>
using namespace std;

int main(){
	//�̹��� ����
	cv::Mat image = cv::imread(fileName);

	//���� üũ
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}
	//�̹��� �����ֱ�
	cv::namedWindow("Hello OpenCV");
	cv::imshow("Hello OpenCV!", image);
	cv::waitKey(0);
}
#endif

#if STEP == 1
//�̹��� �� �� �÷���, Mat���� �Ӽ���
//mat����: https://sites.google.com/site/opencvwikiproject/table-of-contents/opencv-api-reference/core-the-core-functionality/basic-structures/mat
#include "opencv2\highgui.hpp"
#include <iostream>
using namespace std;

int main(){

	//int	flag = CV_LOAD_IMAGE_UNCHANGED;		// =-1. 8bit, color or gray. ���� Į��(Ȥ�� ���) �״��.
	//int	flag = CV_LOAD_IMAGE_GRAYSCALE;			// =0. 8bit, gray 
	int	flag = CV_LOAD_IMAGE_COLOR;			// =1. ?, color. ��� �����̶� 3ä�� Į�� �������� �޾Ƶ���.
	//int	flag = CV_LOAD_IMAGE_ANYDEPTH;		// =2. any depth, ?. ������ ���� depth �״��. 
	//int	flag = CV_LOAD_IMAGE_ANYCOLOR;		// =4. ?, any color
	cout << "flag:" << flag << endl;

	//�̹��� ����
	cv::Mat image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

	//�̹��� �Ӽ�
	cout << "��: " << image.rows << " ��: " << image.cols << endl;
	cout << "����: " << image.size().width << " ����: " << image.size().height << endl;
	cout << "ũ��: " << image.size() << endl;
	cout << "ä��: " << image.channels() << endl;
	cout << "����: " << image.depth() << ", "; //CV_8U�� 0���� define�Ǿ� �־ 0���� ����
	if (image.depth() == CV_8U)
		cout << "8bit unsigned char�Դϴ�." << endl;
	else if (image.depth() == CV_16U)
		cout << "16bit unsigned char�Դϴ�." << endl;
	else if (image.depth() == CV_32F)
		cout << "16bit unsigned float�Դϴ�." << endl;

	//���� üũ
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}

	//�̹��� �����ֱ�
	cv::namedWindow("first");
	cv::imshow("first", image);
	cv::waitKey(0);//0�� ������ ��ٸ� ms������ ŭ Ű ���
}
#endif

#if STEP == 2
#include "opencv2\highgui.hpp"
#include <iostream>
using namespace std;
int main(){

	//�̹��� ����
	cv::Mat image = cv::imread(fileName);

	//���� üũ
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}

	//�̹��� ������
	cv::Mat fliped_0, fliped_pos, fliped_neg;
	cv::flip(image, fliped_0, 0);//�¿� ��Ī
	cv::flip(image, fliped_pos, 1);//���� ��Ī(���)
	cv::flip(image, fliped_neg, -1);//�Ѵ� ��Ī(����)

	//�̹��� �����ֱ�(namedWindow()���̵� imshow�Ҽ� �ִ�!)
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

//���콺 �ݹ�(Ŭ���� �κ��� ���� ������ �˷��ش�.)�� �̹����� �׸��׸���, ���� �����ϱ�
//c++ ĳ��Ʈ �����ڵ� http://prostars.net/65

#include "opencv2\highgui.hpp"
//circle,puttext���� ����ϱ� ���� ��� �߰�
#include "opencv2\imgproc.hpp"
#include <iostream>
using namespace std;

void onMouse(int event, int x, int y, int flags, void *param){
	/*���콺 �ݹ��Լ�: Ŭ���� �κ��� �������� �˷���*/
	cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
	switch (event){
	case CV_EVENT_LBUTTONDOWN:
		std::cout << "at (" << x << "," << y << ") value is "
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << endl;
		break;
	}
}

cv::Mat makeImageBGR(int b = 0, int g = 0, int r = 0) {
	/* BGR ���� �Է¹޾� �ش��ϴ� ������ �����.*/
	cv::Mat ima(300, 240, CV_8UC3, cv::Scalar(b, g, r));
	return ima;
}

int main(){

	//�̹��� ����
	cv::Mat image = cv::imread(fileName);

	//���� üũ
	if (image.empty()){
		cout << "Image Open Failed" << endl;
		exit(0);
	}

	//�̹����� �� �׸��� 
	cv::circle(image,		// ��������
		cv::Point(155, 110),// �߽�
		65,					//������
		255,				//��
		3					//�β�
		);
	cv::putText(image, "Target", cv::Point(100, 200), cv::FONT_HERSHEY_PLAIN, 2.0, 255, 2);

	//���콺 �ݹ� �ޱ�
	cv::namedWindow("target");
	cv::setMouseCallback("target", onMouse, reinterpret_cast<void*>(&image));
	

	//�̹��� �����ֱ�
	cv::imshow("target", image);
	cv::waitKey(0);

	//���� ����&�����ֱ�(�� ������� ����?)
	cv::Mat maden = makeImageBGR(128, 255, 0);
	cv::imshow("maden",maden);
	cv::waitKey(0);
}
#endif

#if STEP == 4
/*
����
1. �־��� 2���� ������ �����
2. ���� ĳ���� �󱼿� ���߾��
3. ��������� ���� ũ���� ������ ������
4. ���ͳݿ��� �簢���� �׸��� �Լ��� ã�Ƽ� ���� �簢���� �׷���
5. �¿�� �������� ���(���) ������ ������
*/
#endif