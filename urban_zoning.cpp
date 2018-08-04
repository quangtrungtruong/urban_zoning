#include "urban_object.h"
#include "stats.h"
#include "data_types.h"
#include "crf/denseho/densecrf.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>

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

#include <zlib.h>

namespace fs = boost::filesystem;

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

UrbanObject::~UrbanObject() {
}

void UrbanObject::Init() {
	srand(2015);		// seed random generator
	param.clear();

	LoadSatelliteGeotagged();

	rows = atoi(param["rows"].c_str());
	cols = atoi(param["cols"].c_str());
	row_4 = rows / 4;
	col_4 = cols / 4;

	VOXEL_PER_METER = atof(param["voxel_per_meter"].c_str());
	METER_PER_VOXEL = 1.f / VOXEL_PER_METER;
	track_threshold = atof(param["track_threshold"].c_str());
	rsme_threshold = atof(param["rsme_threshold"].c_str());

	mrf_rgb_weight = atof(param["mrf_rgb_weight"].c_str());
	mrf_potential = atof(param["mrf_potential"].c_str());
	mrf_iteration = atoi(param["mrf_iteration"].c_str());

	// label constants
	UNLABEL = 0;
	unlabel_color = make_uchar4(128, 128, 128, 255);

	debug_correspondence = false;

	cur_centroid = make_float4(0, 0, 0);
	example_centroid = make_float4(0, 0, 0);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			cur_eigen[i][j] = 0;
			example_eigen[i][j] = 0;
		}
	}

	cout << "Initialization done." << endl;
}


void UrbanObject::LoadSatelliteGeotagged(){
	ifstream satellite_fin(data_dir+"\satellite"+city);
	ifstream geotagged_fin(data_dir + "\geotagged" + city);
	
	if (satellite_fin.is_open())
	{
		satellite_fin >> rows >> cols;
		*satellite_prob = new float4[rows];
		*satellite_zone = new int[rows];
		
		for (int i = 0; i < rows; i++)
		{
			satellite_prob[i] = new float4[cols];
			satellite_zone[i] = new int[cols];
			for (int j = 0; j < cols; j++)
				satellite_fin >> satellite_prob[i][j].x >> satellite_prob[i][j].y >> satellite_prob[i][j].z >> satellite_prob[i][j].w >> satellite_zone[i][j];
		}
		satellite_fin.close();
	}

	pixel_num = rows*cols;

	if (geotagged_fin.is_open())
	{
		geotagged_fin >> num_geotaggeds;
		geotagged_prob = new float4[num_geotaggeds];
		geotagged_info = new int4[num_geotaggeds];
		float attribute;

		for (int i = 0; i < num_geotaggeds; i++)
		{
			geotagged_info[i].w = 0;
			satellite_fin >> geotagged_info[i].x >> geotagged_info[i].y >> geotagged_info[i].x >> geotagged_prob[i].y >> geotagged_prob[i].z >> geotagged_prob[i].w;
			for (int j = 0; j < 255; j++)
				satellite_fin >> attribute;
			satellite_fin >> geotagged_info[i].z;
		}
		satellite_fin.close();
		geotagged_fin.close();
	}
}

////////////////////////////////////////////////////////////////////////////////

int cv_rows, cv_cols;

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

void UrbanObject::RunDenseCRF(bool ho_enabled, bool cooc_enabled) {

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

	 int N = pixel_num;
	 int M = pdf_vec[0].size();
	 std::cout << "Labels: " << M << std::endl;

	 float* unary = new float[N * M];
	 for (int i = 0; i < N; ++i) {
	     for (int k = 0; k < M; ++k) {
	         unary[i * M + k] = -log(pdf_vec[i][k]);
	         if (std::isnan(unary[i * M + k]))
	             unary[i * M + k] = std::numeric_limits<float>::infinity();
	     }
	 }

	 // feature vector for bilateral filtering inside CRF
	 float* gaussian = new float[N * 3];
	 const float sx = 0.1;
	 const float sy = 0.1;
	 const float sz = 0.1;
	 for (int i = 0; i < N; ++i) {
	     gaussian[i * 3 + 0] = pixel_vec[i].x / sx;
	     gaussian[i * 3 + 1] = pixel_vec[i].y / sy;
	     gaussian[i * 3 + 2] = pixel_vec[i].z / sz;
	 }

	DenseCRF crf(N, M);
	 crf.setUnaryEnergy(unary);
	 crf.addPairwiseEnergy(gaussian, 3, 3.0f); // pairwise gaussian
	 crf.addPairwiseEnergy(surface, 6, 5.0f); // pairwise bilateral

	 if (ho_enabled) {
	     crf.setDetHO(1);
	     crf.initMemoryDetHO(0.0005, 1.0);
	     //crf.setDetSegments(proposed_regions);
	 }

	 float* cooc_unary = new float[M];
	 float* cooc_pairwise = new float[M * M];
	 memset(cooc_unary, 0, sizeof(float) * M);
	 memset(cooc_pairwise, 0, sizeof(float) * M * M);
	 if (cooc_enabled) {
	     std::ifstream fin("cooc.txt");
	     for (int i = 0; i < M; ++i)
	         for (int j = 0; j < M; ++j)
	             fin >> cooc_pairwise[i * M + j];
	     for (int i = 0; i < M; ++i)
	         cooc_unary[i] = cooc_pairwise[i * M + i];
	     crf.setHOCooc(1);
	     crf.setCooccurence(cooc_unary, cooc_pairwise, 0.2);
	 }

	 std::clock_t begin = clock();

	 short *map = new short[N];
	 crf.map(10, map);

	 std::clock_t end = clock();
	 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	 std::cout << "CRF " << ho_enabled << " " << cooc_enabled << " time: " << elapsed_secs << std::endl;
}

void UrbanObject::ComputeProbability() {

	std::vector<std::string> prob_list;
	
	int probWidth = cols;
	int probHeight = rows;
	int numClass = 4;

	pdf_vec.resize(pixel_num);
	std::vector<int> confidence(pixel_num, 0);
	confidence.resize(pixel_num);
	for (int i = 0; i < pdf_vec.size(); ++i)
	{
		pdf_vec[i].resize(numClass);
		pdf_vec[i].setZero();
	}

	for (int v = 0; v < pixel_num; ++v) {
		float sum = 0.0f;
		for (int k = 0; k < numClass; ++k) {
			pdf_vec[v][k] += 1.0f;
			sum += pdf_vec[v][k];
		}
		for (int k = 0; k < numClass; ++k) {
			pdf_vec[v][k] /= sum;
		}
	}
}

static bool ReadProbability(float4** prob, int width, int height, int numClass,
	Image<DiscretePdf> &img) {

	// each pixel stores a probability distribution
	img.alloc(width, height);
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			img(x, y).resize(numClass);

	for (int y = 0, i = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x, ++i) {
			img(x, y)[0] = prob[y][x].x;
			img(x, y)[1] = prob[y][x].y;
			img(x, y)[2] = prob[y][x].z;
			img(x, y)[3] = prob[y][x].w;
		}
	}

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float s = img(x, y).sum();
			if (s > 0.0f && s < 0.9f) {
				std::cout << "Pdf not normalized: " << s << std::endl;
			}
			img(x, y).normalize();
		}

	return true;
}