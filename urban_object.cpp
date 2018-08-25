#include "urban_object.h"
#include "proposal.h"
#include "stats.h"
#include <opencv2/opencv.hpp>
#include "crf/denseho/densecrf.h"
#include <fstream>
#include <chrono>  // chrono::system_clock
#include <string>

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

	Preprocess(pdf_vec);
	//Preprocess(nearest_p_vec);

	cout << "Initialization done." << endl;
}


void UrbanObject::LoadSatelliteGeotagged(){
	//string dir_sat = data_dir + "//satellite//" + city + ".txt";
	//string dir_geo = data_dir + "//geotagged//" + city + ".txt";
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
		cout << "Number of geotagged is " << num_geotaggeds << endl;
		geotagged_prob = new float4[num_geotaggeds];
		geotagged_info = new int4[num_geotaggeds];
		std::string delimiter = ",";
		std::string line;
		std::getline(geotagged_fin, line);

		for (int i = 0; i < num_geotaggeds; i++)
		{
			std::getline(geotagged_fin, line);
			size_t pos = 0;
			std::string token;
			double att;
			// replace the position extracted of x,y when loading the geptagged text
			geotagged_info[i].w = 0;
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			geotagged_info[i].y = atoi(token.c_str());
			line.erase(0, pos + delimiter.length());
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			geotagged_info[i].x = atoi(token.c_str());
			line.erase(0, pos + delimiter.length());
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			geotagged_prob[i].x = atof(token.c_str());
			line.erase(0, pos + delimiter.length());
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			geotagged_prob[i].y = atof(token.c_str());
			line.erase(0, pos + delimiter.length());
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			geotagged_prob[i].z = atof(token.c_str());
			line.erase(0, pos + delimiter.length());
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			geotagged_prob[i].w = atof(token.c_str());
			line.erase(0, pos + delimiter.length());

			for (int k = 0; k < 205; k++) {
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				line.erase(0, pos + delimiter.length());
			}

			pos = line.find(delimiter);
			token = line.substr(0, pos);
			line.erase(0, pos + delimiter.length());

			// point out the predicted type of geotaggede photo
			float m = max(max(geotagged_prob[i].x, geotagged_prob[i].y), max(geotagged_prob[i].z, geotagged_prob[i].w));
			if (geotagged_prob[i].x==m)
				geotagged_info[i].z = 0;
			else if (geotagged_prob[i].y == m)
				geotagged_info[i].z = 1;
			else if (geotagged_prob[i].z == m)
				geotagged_info[i].z = 2;
			else 
				geotagged_info[i].z = 3;
		}
		
		cv::Mat satellite_img = cv::imread(dir_sateimg);
		cv::Mat satellite_img_gray;
		int ddepth = CV_16S;
		cv::cvtColor(satellite_img, satellite_img_gray, CV_BGR2GRAY);
		cv::Mat grad_x, grad_y, grad;
		cv::Mat abs_grad_x, abs_grad_y;
		/// Gradient X
		Sobel(satellite_img_gray, grad_x, ddepth, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
		/// Gradient Y
		Sobel(satellite_img_gray, grad_y, ddepth, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		convertScaleAbs(grad_y, abs_grad_y);
		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		pixel_vec.clear();
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				//pixel_vec.push_back(make_float4(satellite_img.at<cv::Vec3b>(i, j).val[0], satellite_img.at<cv::Vec3b>(i, j).val[1], satellite_img.at<cv::Vec3b>(i, j).val[2]));
				pixel_vec.push_back(make_float4(abs_grad_x.at<uchar>(i, j), abs_grad_y.at<uchar>(i, j), abs_grad_y.at<uchar>(i, j) != 0 ? (abs_grad_x.at<uchar>(i, j)*1.0 / abs_grad_y.at<uchar>(i, j)) : 0, satellite_img_gray.at<uchar>(i, j)));

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

	for (int v = 0; v < pixel_num; ++v) {
		float sum = 0.0f;
		for (int k = 0; k < num_class; ++k) {
			nearest_p_vec[v][k] += 1.0f;
			sum += nearest_p_vec[v][k];
		}
		for (int k = 0; k < num_class; ++k) {
			nearest_p_vec[v][k] /= sum;
		}
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

void UrbanObject::Preprocess(std::vector<DiscretePdf> &prop_vec){
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
		prop_vec[v].normalize();
		int index = t_quant(num_class, denominator, prop_vec[v].data());
		auto pdf = prop_vec[v];
		if (!t_reconst(num_class, denominator, index, prop_vec[v].data()))
			printf("Wrong index.\n");

		prop_vec[v].normalize();
		double error = 0.0f;
		for (int k = 0; k < num_class; ++k) {
			xentropy += -pdf[k] * log(fmaxf(prop_vec[v][k], 1e-15));
			error += (pdf[k] - prop_vec[v][k]) * (pdf[k] - prop_vec[v][k]);
		}
		l2 += sqrtf(error);
	}
	printf("Average xentropy error: %lf\n", xentropy / pixel_num);
	printf("Average L2 error: %lf\n", l2 / pixel_num);
	stats.toc("Compression");
}

void UrbanObject::RunDenseCRF(bool ho_enabled, bool pairewise_enabled, double anpha, double beta, int w, int iteration, float gaussian_w, float bilateral_w, float param_w, float weight_term4) {
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
			 double term1 = pdf_vec[i][k] != 0 ? pdf_vec[i][k] : std::numeric_limits< double >::min();
			 double term3 = nearest_p_vec[i][k] != 0 ? 1/nearest_p_vec[i][k] : 1/std::numeric_limits< double >::min();
			 unary[i * M + k] = anpha*(-log(term1)) - beta*log(term3);
			 if (std::isnan(unary[i * M + k]))
				 unary[i * M + k] = std::numeric_limits<float>::infinity();
		 }
	 }

	 // feature vector for bilateral filtering inside CRF

	 float* gaussian = new float[N * 2];
	 const float sx = 1;
	 const float sy = 1;
	 for (int i = 0; i < rows; ++i)
		 for (int j = 0; j < cols; ++j){
		 gaussian[(i * cols + j) * 2 + 0] = i/sx;
		 gaussian[(i * cols + j) * 2 + 1] = j/sy;
	 }

	 float* bilateral = new float[N * 6];
	 const float bsx = 1;
	 const float bsy = 1;
	 const float s = 1;
	 for (int i = 0; i < rows; ++i)
		 for (int j = 0; j < cols; ++j){
			 bilateral[(i * cols + j) * 6 + 0] = i / bsx;
			 bilateral[(i * cols + j) * 6 + 1] = j / bsy;
			 bilateral[(i * cols + j) * 6 + 2] = pixel_vec[i * cols + j].x / s;
			 bilateral[(i * cols + j) * 6 + 3] = pixel_vec[i * cols + j].y / s;
			 bilateral[(i * cols + j) * 6 + 4] = pixel_vec[i * cols + j].w / s;
             bilateral[(i * cols + j) * 6 + 5] = pixel_vec[i * cols + j].z *255;
	 }

	DenseCRF crf(N, M);
	 crf.setUnaryEnergy(unary);

	 if (pairewise_enabled){
		 crf.addPairwiseEnergy(param_w, gaussian, 2, gaussian_w); // pairwise gaussian
         crf.addPairwiseEnergy(bilateral, 6, bilateral_w); // pairwise bilateral
		 //crf.addPairwiseEnergy(param_w);
	 }

	 if (ho_enabled) {
	     crf.setDetHO(1);
	     //crf.initMemoryDetHO(0.0005, 1.0);
         crf.initMemoryDetHO(weight_term4, 0.00005, 1.0);
         //crf.initMemoryDetHO(weight_term4, ho_param1, ho_param2);
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

	 cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255,255,255));
     cv::Mat gt_img(rows, cols, CV_8UC3, cv::Scalar(0,0,0));

	 //evaluate algorithm
     int count = 0, count0 = 0, count1 = 0, count2 = 0, count3 =0;
     int count0_pixel = 0, count1_pixel = 0, count2_pixel = 0, count3_pixel =0;
	 int total = 0, total0 = 0, total1 = 0, total2 = 0, total3 = 0;
	 for (int i = 0; i < rows; i++)
		 for (int j = 0; j < cols; j++){
			 if (real_gt_satellite_img.at<uchar>(i, j) > 30){
				 total++;
				 label_matrix[i][j] = map[i*cols + j] + 1;
                 int gt_label = floor(real_gt_satellite_img.at<uchar>(i, j) / 63 + 0.5);
                 int predict_label = map[i*cols + j] + 1;

                 //img.at<uchar>(i, j) = label_matrix[i][j] * 63;
                 if (label_matrix[i][j]==1)
                     img.at<cv::Vec3b>(i, j) = cv::Vec3b(255,64,0);
                 else if (label_matrix[i][j]==2)
                     img.at<cv::Vec3b>(i, j) = cv::Vec3b(255,255,0);
                 else if (label_matrix[i][j]==3)
                     img.at<cv::Vec3b>(i, j) = cv::Vec3b(64,255,0);
                 else if (label_matrix[i][j]==4)
                     img.at<cv::Vec3b>(i, j) = cv::Vec3b(0,191,255);

                 if (gt_label==1)
                     gt_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255,64,0);
                 else if (gt_label==2)
                     gt_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255,255,0);
                 else if (gt_label==3)
                     gt_img.at<cv::Vec3b>(i, j) = cv::Vec3b(64,255,0);
                 else if (gt_label==4)
                     gt_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0,191,255);

                 if (predict_label==1)
                     total0++;
                 else if (predict_label==2)
                     total1++;
                 else if (predict_label==3)
                     total2++;
                 else if (predict_label==4)
                     total3++;

				 if (predict_label == gt_label)
                 {
                     count++;
                     if (gt_label==1)
                         count0++;
                     else if (gt_label==2)
                         count1++;
                     else if (gt_label==3)
                         count2++;
                     else if (gt_label==4)
                         count3++;
                 }

                 if (gt_label==1)
                     count0_pixel++;
                 else if (gt_label==2)
                     count1_pixel++;
                 else if (gt_label==3)
                     count2_pixel++;
                 else if (gt_label==4)
                     count3_pixel++;
			 }
		 }
    float acc_label0 = count0*1.0/total0;
    float acc_label1 = count1*1.0/total1;
    float acc_label2 = count2*1.0/total2;
    float acc_label3 = count3*1.0/total3;
    cout << "Performance: " << count * 1.0 / total << endl;
    cout << "Class accuracy: " << acc_label0 << ", " << acc_label1 << ", " << acc_label2 << ", " << acc_label3 << ", average class accuracy: " << (acc_label0+acc_label1+acc_label2+acc_label3)/4 << endl;
    //cout << "Percentage of zoning types: " << std::to_string(count0_pixel*1.0/total) << " " << std::to_string(count1_pixel*1.0/total) << " " << std::to_string(count2_pixel*1.0/total) << " " << std::to_string(count3_pixel*1.0/total);
    //cout << "Number of zoning types: " << std::to_string(count0_pixel) << " " << std::to_string(count1_pixel) << " " << std::to_string(count2_pixel) << " " << std::to_string(count3_pixel);
	 string dir_img = data_dir + "//output//" + city + "_crf3_" + std::to_string(ho_enabled) + "_" + std::to_string(pairewise_enabled) + "_" +
		 std::to_string(anpha) + "_" + std::to_string(beta) + "_" + std::to_string(w) + "_" + std::to_string(gaussian_w) + " " + std::to_string(bilateral_w) +
		 " _" + std::to_string(iteration) + "_" + std::to_string(param_w)  + "_" + std::to_string(weight_term4) + "_" + std::to_string(count * 1.0 / total) + ".jpg";
	 imwrite(dir_img, img);
    string dir_gt_img = data_dir + "//output//" + city  + ".jpg";
    imwrite(dir_gt_img, gt_img);

	 acc = count * 1.0 / total;

	 crf.clearMemory();
	 delete[] unary;
	 delete[] gaussian;
	 delete[] bilateral;
	 delete[] map;
}