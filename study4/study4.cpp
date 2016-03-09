#define SUBJECT 0

#if SUBJECT == 12
//필터2d와 sep필터2d사용
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	// Read input image
	cv::Mat image = cv::imread("c:/cmk/miniStudy/Lulu.jpg");
	if (!image.data)		return 0;
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);

	// 
	int ksize = 7;
	int ddepth = -1;
	cv::Mat dst;
	int KernelType = CV_32F;

	// initialize filter details
	int rows = ksize, cols = ksize;
	cv::Mat kernel2D(rows, cols, KernelType);

	for (int j = 0; j<ksize; j++)
		for (int i = 0; i<ksize; i++)
			kernel2D.at<float>(j, i) = 1.0 / ((float)ksize * (float)ksize);

	// 3) Do 2D filtering and show the result ----------------------------------------
	filter2D(image, dst, ddepth, kernel2D);				//, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT )
	cv::namedWindow("Filter2D result"); 	cv::imshow("Filter2D result", dst);

	rows = ksize; cols = 1;
	cv::Mat kernel1D(rows, cols, KernelType);			// create a kernel object.

														// assign the kernel coefficeients  
	for (int j = 0; j<ksize; j++)
		kernel1D.at<float>(j) = 1.0 / ((float)rows);			// 1 dimensional averaging filter. all coefficients have the same value and the sum of them is 1.

	cv::Mat dstH, dstV, dstHV;		// output array
	float delta = 0;

	sepFilter2D(image, dstHV, ddepth, kernel1D, kernel1D, cv::Point(-1, -1), delta);	// internally 1 D filtering twice, in horizontal direction and then in vertical direction 
	sepFilter2D(image, dstH, ddepth, kernel1D, 1, cv::Point(-1, -1), delta);			// Horizontal filtering, FYI.
	sepFilter2D(image, dstV, ddepth, 1, kernel1D, cv::Point(-1, -1), delta);				// Vertical filtering, FYI.
	cv::namedWindow("sepFilter2D-both direction"); 	cv::imshow("sepFilter2D-both direction", dstHV);
	cv::namedWindow("sepFilter2D-x direction"); 	cv::imshow("sepFilter2D-x direction", dstH);				// Horizontal filtering, FYI.
	cv::namedWindow("sepFilter2D-y direction"); 	cv::imshow("sepFilter2D-y direction", dstV);				// Vertical filtering, FYI.
	cv::waitKey();

	//	4) Show the result ------
	sepFilter2D(image, dst, ddepth, kernel1D, 1, cv::Point(-1, -1), delta);			// Horizontal filtering
	sepFilter2D(dst, dst, ddepth, 1, kernel1D, cv::Point(-1, -1), delta);				// Vertical filtering
	cv::namedWindow("Filtering 2 times"); 	cv::imshow("Filtering 2 times", dst);
	cv::waitKey();
	return 0;
}
#endif

#if SUBJECT== 13
//가우시안블러
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
	// Read input image
	cv::Mat image = cv::imread("c:/cmk/miniStudy/Lulu.jpg");		// get source file, 
	if (!image.data)		return 0;
	cv::namedWindow("Input Image");	cv::imshow("Input Image", image);

	// Initialize common variables and objects over the test method sections.
	int ksize = 7;	// Kernel Size. must be odd positive.
	float	sigma = 3.0;
	cv::Mat dst;
	int ddepth = -1;			// src와 같은 type의 dst 결과를 지정.
	double delta = 0;			// 처리하고 난 결과에 delta를 가산함.


								///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
								// Method 1 - Use GaussianBlur function for Gaussian blurring
								///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::GaussianBlur(image, dst, cv::Size(ksize, ksize), sigma);		// kernel size = ksize
	cv::namedWindow("M1-GaussianBlur");		cv::imshow("M1-GaussianBlur", dst); cv::waitKey();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Method 2 - Get a Gaussian Kernel using getGaussianKernel function and apply it to sepFilter2D function for Gaussian blurring
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat GaussKernel1D = cv::getGaussianKernel(ksize, sigma, CV_32F);		// returns ksize *1 filter coefficients 

	cv::Mat_<float>::const_iterator it = GaussKernel1D.begin<float>();
	cv::Mat_<float>::const_iterator itend = GaussKernel1D.end<float>();

	printf("\n1D Gaussian Kernel Matrix : %d X % d", GaussKernel1D.rows, GaussKernel1D.cols);
	for (int i = 0; it != itend; ++it) {
		if (i++ % ksize == 0) 		std::cout << std::endl;
		printf("  %4.2f", *it);
	}

	sepFilter2D(image, dst, ddepth, GaussKernel1D, GaussKernel1D, cv::Point(-1, -1), delta);
	cv::namedWindow("M2-sepFilter2D"); 	cv::imshow("M2-sepFilter2D", dst);

	// 양방향 처리 결과와 가로 방향 처리하고, 그 결과를 세로 방향 처리한 결과가 같다. 즉, linearly seperable하다.
	sepFilter2D(image, dst, ddepth, GaussKernel1D, 1, cv::Point(-1, -1), delta);			// 가로 방향블러링
	sepFilter2D(dst, dst, ddepth, 1, GaussKernel1D, cv::Point(-1, -1), delta);				// 세로 방향 블러링
	cv::namedWindow("M2-sepFilter2D 2 times"); 	cv::imshow("M2-sepFilter2D 2 times", dst);
	cv::waitKey();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Method 3 - Compute a 2D Gaussian Kernel yourself and apply it to Filter2D function for Gaussian blurring
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cv::Mat Kernel2D(ksize, ksize, CV_32F);		// Define a 2D kernel

	it = Kernel2D.begin<float>();
	itend = Kernel2D.end<float>();

	// assign the kernel coefficients  
	int		iHalfMaskSize = ksize / 2;
	float	fSum = 0, fTmp1, fTmp2;
	float	fSigma = (float)sigma;

	for (int y = -iHalfMaskSize; y <= iHalfMaskSize; y++)
		for (int x = -iHalfMaskSize; x <= iHalfMaskSize; x++) {
			fTmp1 = -((float)x*x + (float)y*y);
			fTmp2 = fTmp1 / (2 * fSigma*fSigma);
			Kernel2D.at<float>(y + iHalfMaskSize, x + iHalfMaskSize) = (float)exp(fTmp2);
			fSum += (float)exp(fTmp2);								// for nomalization of each coefficient  
		}

	// Kernel[][]에 있는 계수를 정규화한다. 
	//for (  ; it!= itend; ++it)		*it = fSum/ *it;
	for (int y = -iHalfMaskSize; y <= iHalfMaskSize; y++)
		for (int x = -iHalfMaskSize; x <= iHalfMaskSize; x++) {
			Kernel2D.at<float>(y + iHalfMaskSize, x + iHalfMaskSize) /= fSum;
		}

	printf("\n2D Gaussian Blurring Coefficients Matrix : %d X % d", ksize, ksize);
	for (int i = 0; it != itend; ++it) {
		if (i++ % ksize == 0) 		std::cout << std::endl;
		printf("  %4.2f", *it);
	}

	//	Do filtering ----------------------------------------------
	filter2D(image, dst, ddepth, Kernel2D);				//, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT )
	cv::namedWindow("M3-filter2D 2D Kernel Computed"); 	cv::imshow("M3-filter2D 2D Kernel Computed", dst);
	cv::waitKey();


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Method 4 - Compute a 1D Gaussian Kernel yourself and apply it to sepFilter2D function for Gaussian blurring
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cv::Mat Kernel1D(ksize, 1, CV_32F);		// Define a 1D kernel

	it = Kernel1D.begin<float>();
	itend = Kernel1D.end<float>();

	// assign the kernel coefficients  
	//int		iHalfMaskSize = ksize / 2;
	//float	fSum=0, fTmp1, fTmp2;
	//float	fSigma = (float) sigma;

	fSum = 0;			// initialize accumulator
	for (int y = -iHalfMaskSize; y <= iHalfMaskSize; y++) {
		fTmp1 = -((float)y*y);
		fTmp2 = fTmp1 / (2 * fSigma*fSigma);
		Kernel1D.at<float>(y + iHalfMaskSize) = (float)exp(fTmp2);
		fSum += (float)exp(fTmp2);								// for nomalization of each coefficient
	}

	// Kernel[][]에 있는 계수를 정규화한다. 
	//for (  ; it!= itend; ++it)		*it = fSum/ *it;
	for (int y = -iHalfMaskSize; y <= iHalfMaskSize; y++)
		Kernel1D.at<float>(y + iHalfMaskSize) /= fSum;

	printf("\n1D Gaussian Blurring Coefficients Matrix : %d X % d", ksize, ksize);
	for (int i = 0; it != itend; ++it) {
		if (i++ % ksize == 0) 		std::cout << std::endl;
		printf("  %4.2f", *it);
	}
	printf("\n", *it);

	//	Do filtering ----------------------------------------------
	sepFilter2D(image, dst, ddepth, Kernel1D, Kernel1D, cv::Point(-1, -1), delta);
	cv::namedWindow("M4-SepFilter2D 1D Kernel Computed");
	cv::imshow("M4-SepFilter2D 1D Kernel Computed", dst);
	cv::waitKey();  return 0;
}
#endif