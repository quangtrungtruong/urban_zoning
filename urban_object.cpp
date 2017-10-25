#include "urban_object.h"
#include "proposal.h"
#include "stats.h"
#include <opencv2/opencv.hpp>
#include "crf/denseho/densecrf.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <random> // necessary for generation of random floats (for SSAO sample kernel and noise texture)

static std::string display_mode_string[UrbanObject::DisplayMode::NUM_DISPLAY_MODES] = {
	"Probability",
	"Segmentation",
	"NYU class",
	"Graphcut",
	"MRF",
	"Texture",
	"Grey",
	"Cloud",
	"AABB",
	"OBB",
	"Image based",
	"Reflectance",
	"Shading"
};

UrbanObject::UrbanObject(){}

UrbanObject::UrbanObject(const char* data_dir, const char* city) {
	this->data_dir = data_dir;
	this->city = city;

	xy8[0] = make_int2(-1, 0);
	xy8[1] = make_int2(1, 0);
	xy8[2] = make_int2(0, -1);
	xy8[3] = make_int2(0, 1);
	xy8[4] = make_int2(-1, -1);
	xy8[5] = make_int2(-1, 1);
	xy8[6] = make_int2(1, -1);
	xy8[7] = make_int2(1, 1);

	Init();
}

UrbanObject::~UrbanObject() {
}

void UrbanObject::Init() {
	LoadSatelliteGeotagged();
	ComputeProbability();
	ConvertNearestZoneTypeVec();
	cout << "Initialization done." << endl;
}


void UrbanObject::LoadSatelliteGeotagged(){
	string dir_sat = data_dir + "//satellite//" + city + ".txt";
	string dir_geo = data_dir + "//geotagged//" + city + ".txt";
	string dir_sateimg = data_dir + "//satellite//" + city + ".jpg";
	string dir_distance_tran = data_dir + "//distance_transform//" + city + "_dt_";
	ifstream satellite_fin(dir_sat.c_str());
	ifstream geotagged_fin(dir_geo.c_str());
	ifstream dist_t_fin0((dir_distance_tran + "0.txt").c_str());
	ifstream dist_t_fin1((dir_distance_tran + "1.txt").c_str());
	ifstream dist_t_fin2((dir_distance_tran + "2.txt").c_str());
	ifstream dist_t_fin3((dir_distance_tran + "3.txt").c_str());

	cout << "Start with city " << city << endl;
	
	int temp;
	if (satellite_fin.is_open())
	{
        satellite_fin >> rows >> cols;
		dist_t_fin0 >> temp >> temp;
		dist_t_fin1 >> temp >> temp;
		dist_t_fin2 >> temp >> temp;
		dist_t_fin3 >> temp >> temp;

        satellite_prob = new float4*[rows];
		nearest_distance = new float4*[rows];
		satellite_zone = new int*[rows];
		label_matrix = new int*[rows];
		for (int i = 0; i < rows; i++)
		{
			satellite_prob[i] = new float4[cols];
			nearest_distance[i] = new float4[cols];
			satellite_zone[i] = new int[cols];
			label_matrix[i] = new int[cols];
			for (int j = 0; j < cols; j++)
			{ 
				satellite_fin >> satellite_prob[i][j].x >> satellite_prob[i][j].y >> satellite_prob[i][j].z >> satellite_prob[i][j].w >> satellite_zone[i][j];
				dist_t_fin0 >> nearest_distance[i][j].x; dist_t_fin1 >> nearest_distance[i][j].y; dist_t_fin2 >> nearest_distance[i][j].z; dist_t_fin3 >> nearest_distance[i][j].w;
			}
		}
		satellite_fin.close();
		dist_t_fin0.close();
		dist_t_fin1.close();
		dist_t_fin2.close();
		dist_t_fin3.close();
	}

	pixel_num = rows*cols;

	if (geotagged_fin.is_open())
	{
		geotagged_fin >> num_geotaggeds;
		geotagged_prob = new float4[num_geotaggeds];
		geotagged_info = new int4[num_geotaggeds];
		std::string delimiter = ",";
		if (geotagged_fin.is_open())
		{
			geotagged_fin >> num_geotaggeds;
			geotagged_prob = new float4[num_geotaggeds];
			geotagged_info = new int4[num_geotaggeds];
			std::string line;
			std::getline(geotagged_fin, line);

			for (int i = 0; i < num_geotaggeds; i++)
			{
				std::getline(geotagged_fin, line);
				size_t pos = 0;
				std::string token;
				double att;
				geotagged_info[i].w = 0;

				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_info[i].x = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_info[i].y = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_prob[i].x = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_prob[i].y = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_prob[i].z = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_prob[i].w = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());

				for (int k = 0; k < 205; k++) {
					pos = line.find(delimiter);
					token = line.substr(0, pos);
					line.erase(0, pos + delimiter.length());
				}

				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_info[i].z = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
			}
		}
		
		cv::Mat satellite_img = cv::imread(dir_sateimg);

		pixel_vec.clear();
		for (int i = 0; i < cols; i++) {
			for (int j = 0; j < rows; j++) 
			{ 
				pixel_vec.push_back(make_float4(satellite_img.at<cv::Vec3b>(j, i)[0], satellite_img.at<cv::Vec3b>(j, i)[1], satellite_img.at<cv::Vec3b>(j, i)[2]));
			}
				
		}

		satellite_fin.close();
		geotagged_fin.close();
	}
}

void UrbanObject::ConvertNearestZoneTypeVec(){
	nearest_p_vec.resize(pixel_num);
	for (int i = 0; i < pdf_vec.size(); ++i)
	{
		nearest_p_vec[i].resize(num_class);
		nearest_p_vec[i].setZero();
	}

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++){
			nearest_p_vec[i*cols + j][0] = nearest_distance[i][j].x;
			nearest_p_vec[i*cols + j][1] = nearest_distance[i][j].y;
			nearest_p_vec[i*cols + j][2] = nearest_distance[i][j].z;
			nearest_p_vec[i*cols + j][3] = nearest_distance[i][j].w;
		}
}

void UrbanObject::ComputeProbability() {
	pdf_vec.resize(pixel_num);
	std::vector<int> confidence(pixel_num, 0);
	confidence.resize(pixel_num);
	for (int i = 0; i < pdf_vec.size(); ++i)
	{
		pdf_vec[i].resize(num_class);
		pdf_vec[i].setZero();
	}

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++){
			pdf_vec[i*cols + j][0] = satellite_prob[i][j].x;
			pdf_vec[i*cols + j][1] = satellite_prob[i][j].y;
			pdf_vec[i*cols + j][2] = satellite_prob[i][j].z;
			pdf_vec[i*cols + j][3] = satellite_prob[i][j].w;
		}


	for (int v = 0; v < pixel_num; ++v) {
		float sum = 0.0f;
		for (int k = 0; k < num_class; ++k) {
			pdf_vec[v][k] += 1.0f;
			sum += pdf_vec[v][k];
		}
		for (int k = 0; k < num_class; ++k) {
			pdf_vec[v][k] /= sum;
		}
	}
}

#define MAX_DIMENSIONS 100
#define DENOMINATOR 100
static int* binomial[DENOMINATOR + 1];
static int coefficients[(DENOMINATOR + 1) * (DENOMINATOR + 2) / 2];

typedef struct { float delta; int index; } pdist;

int pdistcmp(const void* a, const void* b)
{
	pdist* p = (pdist*)a;
	pdist* q = (pdist*)b;
	if (p->delta > q->delta) return 1;
	if (p->delta < q->delta) return -1;
	return 0;
}

int t_points(int m, int n) { return binomial[n + m - 1][m - 1]; }
int t_quant(int m, int n, float* p)
{
	int k[MAX_DIMENSIONS];
	pdist dists[MAX_DIMENSIONS];

	int sum = 0;
	for (int i = 0; i < m; ++i) {
		k[i] = (int)floor(p[i] * n + 0.5);
		sum += k[i];
	}

	int delta = sum - n;
	if (delta) {
		for (int i = 0; i < m; ++i) {
			dists[i].delta = (float)k[i] - p[i] * n;
			dists[i].index = i;
		}
		std::qsort(dists, m, sizeof(pdist), pdistcmp);
		if (delta > 0) {
			for (int j = m - delta; j < m; ++j)
				k[dists[j].index]--;
		}
		else {
			for (int j = 0; j < -delta; ++j)
				k[dists[j].index]++;
		}
	}

	int index = 0;
	for (int i = 0; i < m - 2; ++i) {
		int s = 0;
		for (int j = 0; j < k[i]; ++j)
			s += binomial[n - j + m - i - 2][m - i - 2];
		index += s;
		n -= k[i];
	}
	index += k[m - 2];
	return index;
}

int t_reconst(int m, int n, int index, float* p)
{
	int k[MAX_DIMENSIONS];
	float n_inv = 1.0f / (float)n;

	if (index < 0 || index >= t_points(m, n))
		return 0;

	for (int i = 0; i < m - 2; ++i) {
		int s = 0;
		int j;
		for (j = 0; j < n; ++j) {
			int x = binomial[n - j + m - i - 2][m - i - 2];
			if (index - s < x) break;
			s += x;
		}
		k[i] = j;
		index -= s;
		n -= j;
	}
	k[m - 2] = index;
	k[m - 1] = n - index;
	for (int j = 0; j < m; ++j)
		p[j] = (float)k[j] * n_inv;
	return 1;
}

void UrbanObject::RunDenseCRF(bool ho_enabled, bool pairewise_enabled, double anpha, double beta, int w, int iteration) {
	int* b = coefficients;
	for (int n = 0; n <= DENOMINATOR; ++n) {
		binomial[n] = b;
		b += n + 1;
		binomial[n][0] = binomial[n][n] = 1;
		for (int k = 1; k < n; ++k)
			binomial[n][k] = binomial[n - 1][k - 1] + binomial[n - 1][k];
	}

	int denominator = 8;
	printf("Bits required: %f\n", log2(t_points(num_class, denominator)));
	double l2 = 0.0f;
	double xentropy = 0.0f;
	Stats stats;
	stats.tic();
	for (int v = 0; v < pixel_num; ++v) {
		pdf_vec[v].normalize();
		int index = t_quant(num_class, denominator, pdf_vec[v].data());
		auto pdf = pdf_vec[v];
		if (!t_reconst(num_class, denominator, index, pdf_vec[v].data()))
			printf("Wrong index.\n");

		 pdf_vec[v].normalize();
		 double error = 0.0f;
		 for (int k = 0; k < num_class; ++k) {
		     xentropy += -pdf[k] * log(fmaxf(pdf_vec[v][k], 1e-15));
		     error += (pdf[k] - pdf_vec[v][k]) * (pdf[k] - pdf_vec[v][k]);
		 }
		 l2 += sqrtf(error);
	}
	printf("Average xentropy error: %lf\n", xentropy / pixel_num);
	printf("Average L2 error: %lf\n", l2 / pixel_num);
	stats.toc("Compression");

	 std::vector<region> proposed_regions;
	 if (ho_enabled) {
		 propose_regions_fast(geotagged_info, num_geotaggeds, rows, cols, w, proposed_regions);
	 }

	 int N = pixel_num;
	 int M = pdf_vec[0].size();
	 std::cout << "Labels: " << M << std::endl;

	 float* unary = new float[N * M];
	 for (int i = 0; i < N; ++i) {
		 for (int k = 0; k < M; ++k) {
			 unary[i * M + k] = anpha*(-log(pdf_vec[i][k])) + beta*log(1 / nearest_p_vec[i][k]);
			 if (std::isnan(unary[i * M + k]))
				 unary[i * M + k] = std::numeric_limits<float>::infinity();
		 }
	 }

	 // feature vector for bilateral filtering inside CRF

	 float* gaussian = new float[N * 2];
	 const float sx = 3;
	 const float sy = 3;
	 for (int i = 0; i < rows; ++i)
		 for (int j = 0; j < cols; ++j){
		 gaussian[(i * cols + j) * 2 + 0] = i / sx;
		 gaussian[(i * cols + j) * 2 + 1] = j / sy;
	 }

	 float* bilateral = new float[N * 5];
	 const float bsx = 50;
	 const float bsy = 50;
	 const float sr = 15;
	 const float sg = 15;
	 const float sb = 15;
	 for (int i = 0; i < rows; ++i)
		 for (int j = 0; j < cols; ++j){
			 bilateral[(i * cols + j) * 5 + 0] = i / bsx;
			 bilateral[(i * cols + j) * 5 + 1] = j / bsy;
			 bilateral[(i * cols + j) * 5 + 2] = pixel_vec[i * cols + j].x / sr;
			 bilateral[(i * cols + j) * 5 + 3] = pixel_vec[i * cols + j].y / sg;
			 bilateral[(i * cols + j) * 5 + 4] = pixel_vec[i * cols + j].z / sb;
	 }

	DenseCRF crf(N, M);
	 crf.setUnaryEnergy(unary);

	 if (pairewise_enabled){
		 crf.addPairwiseEnergy(gaussian, 2, 3.0f); // pairwise gaussian
		 crf.addPairwiseEnergy(bilateral, 5, 4.0f); // pairwise bilateral
	 }

	 if (ho_enabled) {
	     crf.setDetHO(1);
	     crf.initMemoryDetHO(0.0005, 1.0);
	     crf.setDetSegments(proposed_regions);
	 }

	 std::clock_t begin = clock();

	 short *map = new short[N];
	 crf.map(iteration, map);

	 std::clock_t end = clock();
	 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	 // load real gt
	 string dir_gt_sateimg = data_dir + "//gt//" + city + ".jpg";
	 cv::Mat real_gt_satellite_img = cv::imread(dir_gt_sateimg, 0);

	 cv::Mat img(rows, cols, CV_8UC1, cv::Scalar(0));

	 //evaluate algorithm
	 int count = 0;
	 int total = 0;
	 for (int i = 0; i < rows; i++)
		 for (int j = 0; j < cols; j++){
			 if (real_gt_satellite_img.at<uchar>(i, j) > 30){
				 total++;
				 label_matrix[i][j] = map[i*cols + j] + 1;
				 img.at<uchar>(i, j) = label_matrix[i][j] * 63;
				 if ((img.at<uchar>(i, j) / 63) == floor(real_gt_satellite_img.at<uchar>(i, j) / 63 + 0.5))
					 count++;
			 }
		 }
	 cout << "Performance: " << count * 1.0 / total;

	 string dir_img = data_dir + "//output//" + city + "_crf3_" + std::to_string(ho_enabled) + "_" + std::to_string(pairewise_enabled) + "_" + 
		 std::to_string(anpha) + "_" + std::to_string(beta) + "_" + std::to_string(w) + " _" + std::to_string(iteration) + "_" + std::to_string(count * 1.0 / total) + ".jpg";
	 imwrite(dir_img, img);

	 crf.clearMemory();
	 delete[] unary;
	 delete[] gaussian;
	 delete[] bilateral;
	 delete[] map;
}

void UrbanObject::GenerateText(string data_dir, string city){
	//cout << "do generation";
	//string dir0 = data_dir + "//DNN//" + city + "//1.jpg";
	//string dir1 = data_dir + "//DNN//" + city + "//2.jpg";
	//string dir2 = data_dir + "//DNN//" + city + "//3.jpg";
	//string dir3 = data_dir + "//DNN//" + city + "//4.jpg";
	//string gt_dir = data_dir + "//gt//" + city + ".jpg";
	//cv::Mat img0 = cv::imread(dir0, 0);
	//cv::Mat img1 = cv::imread(dir1, 0);
	//cv::Mat img2 = cv::imread(dir2, 0);
	//cv::Mat img3 = cv::imread(dir3, 0);
	//cv::Mat gt = cv::imread(gt_dir, 0);
	////// save to distance transform to text file
	//ofstream infile((data_dir + "//" + city + "_regen.txt"));
	//int rows = img0.rows;
	//int cols = img0.cols;
	//infile << rows << " " << cols << endl;
	//for (int i = 0; i < rows; i++) {
	//	for (int j = 0; j < cols; j++)
	//	{
	//		if (gt.at<uchar>(i, j) >30){
	//			double sum = img0.at<uchar>(i, j) + img1.at<uchar>(i, j) + img2.at<uchar>(i, j) + img3.at<uchar>(i, j);
	//			infile << img0.at<uchar>(i, j) / sum << " " << img1.at<uchar>(i, j) / sum
	//				<< " " << img2.at<uchar>(i, j) / sum << " " << img3.at<uchar>(i, j) / sum << " " << floor(gt.at<uchar>(i, j) / 64 + 0.5) << " ";
	//		}
	//		else
	//			infile << "0 0 0 0 0 ";
	//	}
	//	infile << endl;
	//}
	//infile.close();

	cout << "do generation";
	string dir0 = data_dir + "//DNN//" + city + "//1.jpg";
	string dir1 = data_dir + "//DNN//" + city + "//2.jpg";
	string dir2 = data_dir + "//DNN//" + city + "//3.jpg";
	string dir3 = data_dir + "//DNN//" + city + "//4.jpg";
	string gt = data_dir + "//gt//" + city + ".jpg";
	cv::Mat gt_img = cv::imread(gt, 0);
	cv::Mat img0 = cv::imread(dir0, 0);
	cv::Mat img1 = cv::imread(dir1, 0);
	cv::Mat img2 = cv::imread(dir2, 0);
	cv::Mat img3 = cv::imread(dir3, 0);
	int rows = img0.rows;
	int cols = img0.cols;
	int count = 0;
	int total = 0;
	cv::Mat prop_img(rows, cols, CV_8UC1, cv::Scalar(0));
	// save to distance transform to text file
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++)
		{
			if (gt_img.at<uchar>(i, j) >30){
				total++;
				int m = max(max(img0.at<uchar>(i, j), img1.at<uchar>(i, j)), max(img2.at<uchar>(i, j), img3.at<uchar>(i, j)));
				if (img0.at<uchar>(i, j) == m)
					prop_img.at<uchar>(i, j) = 60;
				else if (img1.at<uchar>(i, j) == m)
					prop_img.at<uchar>(i, j) = 120;
				else if (img2.at<uchar>(i, j) == m)
					prop_img.at<uchar>(i, j) = 180;
				else
					if (img3.at<uchar>(i, j) == m)
						prop_img.at<uchar>(i, j) = 240;
				int a = prop_img.at<uchar>(i, j);
				int b = gt_img.at<uchar>(i, j);
				int c = floor(gt_img.at<uchar>(i, j) / 64 + 0.5);

				if ((prop_img.at<uchar>(i, j) / 60) == floor(gt_img.at<uchar>(i, j) / 64 + 0.5))
					count++;
			}
		}
	}
	cout << "Accuracy: " << count*1.0 / total;
	string prop_dir_img = data_dir + "//output//prop_4imgs_" + city + ".jpg";
	imwrite(prop_dir_img, prop_img);
}