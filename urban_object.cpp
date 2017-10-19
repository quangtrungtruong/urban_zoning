#include "urban_object.h"
#include "stats.h"
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

UrbanObject::UrbanObject(const char* data_dir, const char* city) {
	/*
	string str = argv;
	if (str.find("/") == string::npos) {
	// input is a name of the dataset
	dataset = argv;
	}
	else {

	// input is a path to the dataset folder or a mesh file in the dataset folder
	data_dir = argv;

	fs::path path(data_dir);
	if (fs::is_directory(path) == false) {
	mesh_file = path.filename().string();
	data_dir = path.remove_filename().string();
	}

	if (data_dir.back() != '/') data_dir += "/";
	size_t last = data_dir.length() - 1;
	size_t pos = data_dir.rfind("/", data_dir.length() - 2);

	if (pos == string::npos) {
	throw std::runtime_error("Invalid input path");
	}

	dataset = data_dir.substr(pos + 1, last - pos - 1);
	std::cout << "Dataset: " << dataset << std::endl;

	if (mesh_file == "") {
	mesh_file = dataset + ".ply";
	}
	}
	*/
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

	cout << "Initialization done." << endl;
}


void UrbanObject::LoadSatelliteGeotagged(){
	string dir_sat = data_dir + "//satellite//" + city + ".txt";
	string dir_geo = data_dir + "//geotagged//" + city + ".txt";
	ifstream satellite_fin(dir_sat.c_str());
	ifstream geotagged_fin(dir_geo.c_str());

	cout << "Start with city " << city;
	
	if (satellite_fin.is_open())
	{
        satellite_fin >> rows >> cols;
        satellite_prob = new float4*[rows];
		satellite_zone = new int*[rows];
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
				geotagged_info[i].x = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_info[i].y = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_info[i].z = atoi(token.c_str());
				line.erase(0, pos + delimiter.length());
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				geotagged_info[i].w = atoi(token.c_str());
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
		satellite_fin.close();
		geotagged_fin.close();
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


//void UrbanObject::ReSegment(uint root_vertex, uint tail_vertex) {
//	// assume root and tail vertex has the same label
//	uint label = label_vec[root_vertex];
//
//	vector<uint> curr_vec;
//	vector<int> curr_map(vertex_num);         // map vertex to local index in order to build subgraph
//	int k = 0;
//	for (uint i = 0; i < vertex_num; ++i) {
//		if (label_vec[i] == label) {
//			curr_vec.push_back(i);
//			curr_map[i] = k;
//			k++;
//		}
//		else {
//			curr_map[i] = -1;
//		}
//	}
//
//	set<pii> curr_edge;
//	/*// V^2 E complexity, too slow when need to split a big label with ~500K points
//	for (uint i = 0; i < curr_vec.size(); ++i) {
//	for (uint j = i; j < curr_vec.size(); ++j) {
//	set<pii>::iterator it = eset.find(pii(curr_vec[i], curr_vec[j]));
//	if (it != eset.end())
//	curr_edge.insert(pii(i, j));
//	}
//	}*/
//	for (set<pii>::iterator it = eset.begin(); it != eset.end(); ++it) {
//		pii edge = *it;
//
//		if (curr_map[edge.first] < 0 || curr_map[edge.second] < 0)
//			continue;
//		int i = curr_map[edge.first];
//		int j = curr_map[edge.second];
//		if (i < j)
//			curr_edge.insert(pii(i, j));
//		else
//			curr_edge.insert(pii(j, i));
//	}
//
//	vector<uint> curr_label(curr_vec.size());
//	vector<float4> curr_normal(curr_vec.size());
//	vector<uchar4> curr_rgb(curr_vec.size());
//	for (uint i = 0; i < curr_vec.size(); ++i) {
//		curr_normal[i] = normal_vec[curr_vec[i]];
//		curr_rgb[i] = rgb_vec[curr_vec[i]];
//	}
//	SegmentMesh sm;
//	sm.ReCompute(curr_edge, mesh_vertex_threshold2, curr_normal, curr_rgb, curr_label);
//
//	map<uint, uint> lmap;
//	map<uint, uchar4> cmap;     // color map
//	for (uint i = 0; i < curr_label.size(); ++i) {
//		map<uint, uint>::iterator it = lmap.find(curr_label[i]);
//		if (it == lmap.end()) {
//			lmap.insert(make_pair(curr_label[i], curr_vec[i] + LABEL_BASE));
//			cmap.insert(make_pair(curr_vec[i] + LABEL_BASE, RandomColor()));
//		}
//	}
//
//	if (lmap.size() > 1) {
//		undo_color.Insert(color_vec);
//		undo_label.Insert(label_vec);
//		if (anno_window)
//			anno_window->pushState();
//
//		for (uint i = 0; i < curr_vec.size(); ++i) {
//			label_vec[curr_vec[i]] = lmap[curr_label[i]];
//			color_vec[curr_vec[i]] = cmap[label_vec[curr_vec[i]]];
//		}
//	}
//
//	if (label_vec[root_vertex] == label_vec[tail_vertex]) {
//		cout << "Warning: re-segmentation still assigns same label to root and tail vertex." << endl;
//		return;
//	}
//
//
//	// doing graph cut alone is not enough. It will results in many small labels that needs to be merged.
//	// try run another MRF on this subgraph?
//	// we set the neighbor cost of the region of start and end vertex to inifinity
//	// to prevent them from merging again
//	MRFConstraint(curr_vec, curr_map, cmap, root_vertex, tail_vertex);
//
//	if (anno_window)
//		anno_window->setUpdateRequired(true);
//
//	count_resegment++;
//	undo_count.Insert(make_uint3(count_merge, count_resegment, count_graphcut));
//}
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

	/* std::vector<region> proposed_regions;
	 if (ho_enabled) {
	     propose_regions_fast(vertex_vec, normal_vec, eset, proposed_regions);
	 }*/

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
	 /*float* gaussian = new float[N * 3];
	 const float sx = 0.1;
	 const float sy = 0.1;
	 const float sz = 0.1;
	 for (int i = 0; i < N; ++i) {
	     gaussian[i * 3 + 0] = pixel_vec[i].x / sx;
	     gaussian[i * 3 + 1] = pixel_vec[i].y / sy;
	     gaussian[i * 3 + 2] = pixel_vec[i].z / sz;
	 }*/

	/* float* surface = new float[N * 6];
	 const float snx = 0.1;
	 const float sny = 0.1;
	 const float snz = 0.1;
	 for (int i = 0; i < N; ++i) {
	     surface[i * 6 + 0] = vertex_vec[i].x / sx;
	     surface[i * 6 + 1] = vertex_vec[i].y / sy;
	     surface[i * 6 + 2] = vertex_vec[i].z / sz;
	     surface[i * 6 + 3] = normal_vec[i].x / snx;
	     surface[i * 6 + 4] = normal_vec[i].y / sny;
	     surface[i * 6 + 5] = normal_vec[i].z / snz;
	 }*/

	DenseCRF crf(N, M);
	 crf.setUnaryEnergy(unary);
	 //crf.addPairwiseEnergy(gaussian, 3, 3.0f); // pairwise gaussian
	 //crf.addPairwiseEnergy(surface, 6, 5.0f); // pairwise bilateral

	 //if (ho_enabled) {
	 //    crf.setDetHO(1);
	 //    crf.initMemoryDetHO(0.0005, 1.0);
	 //    crf.setDetSegments(proposed_regions);
	 //}

	 // need to implement this section to load cooc_unary and cooc_pairwise
	 // it is loaded in original author code of higherorder.cpp
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

	 for (int i = 0; i < N; ++i) {
	     label_vec[i] = map[i] + 1;
	 }

	 crf.clearMemory();
	 delete[] unary;
	 /*delete[] gaussian;
	 delete[] surface;*/
	 delete[] map;
}

//void UrbanObject::RunDenseCRF(bool ho_enabled, bool cooc_enabled) {
//	int num_of_labels = 4;
//
//	//cout << "solving image " << files << ": " << dataset->testImageFiles[files] << endl;
//
//	short * map = new short[rows*cols];
//	DenseCRF2D crf_plane(cols, rows, num_of_labels);
//
//	float* unary = new float[rows*cols];
//	//N: height, rows
//	//M: width, cols
//	for (int i = 0; i < rows; ++i) {
//		for (int k = 0; k < cols; ++k) {
//		    unary[i * cols + k] = -log(pdf_vec[i][k]);
//		    if (std::isnan(unary[i * cols + k]))
//		        unary[i * cols + k] = std::numeric_limits<float>::infinity();
//		}
//	}
//
//	im_orig = new unsigned char[3 * rows*cols];
//
//	unsigned char *im = rgbImage.GetData();
//	long offset = 3 * rows*cols;
//
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 0; j < cols; j++)
//		{
//			im_orig[offset + 3 * ((rows - (i + 1))*cols + j) + 0] = im[3 * (i*cols + j) + 0];
//			im_orig[offset + 3 * ((rows - (i + 1))*cols + j) + 1] = im[3 * (i*cols + j) + 1];
//			im_orig[offset + 3 * ((rows - (i + 1))*cols + j) + 2] = im[3 * (i*cols + j) + 2];
//		}
//	}
//
//
//	// unary
//	crf_plane.setUnaryEnergy(unary);
//
//	// pairwise
//	crf_plane.addPairwiseGaussian(3, 3, 3);
//	crf_plane.addPairwiseBilateral(50, 50, 15, 15, 15, im_orig, 5);
//
//	int ho_on = 0;
//	int ho_det = 0;
//	int ho_cooc = 0;
//
//	//// set PN potts ho_order
//	// ho_on = 1;
//	crf_plane.set_ho(ho_on);
//	if (ho_on) {
//		set_ho_layers();
//		crf_plane.ho_mem_init(imagewidth, imageheight, layers_dir, num_of_layers, ho_stats_pot, ho_seg_ext, ho_sts_ext, 0.0006, 1.0);
//		crf_plane.readSegments(dataset->testImageFiles[files]);
//	}
//
//	////set ho_det
//	// ho_det = 1;
//	crf_plane.set_hodet(ho_det);
//	if (ho_det) {
//		set_det_layers();
//		crf_plane.det_ho_mem_init(imagewidth, imageheight, det_seg_dir, det_bb_dir, det_seg_ext, det_bb_ext, 0.00005, 1.0);
//		crf_plane.det_readSegmentIndex(dataset->testImageFiles[files]);
//	}
//
//	//// cooccurrence
//	// ho_cooc = 1;
//	crf_plane.set_hocooc(ho_cooc);
//	if (ho_cooc) {
//		crf_plane.setco_occurrence(cooc_unary, cooc_pairwise, 10.0);
//	}
//
//	// start inference 
//	clock_t start = clock();
//	crf_plane.map(5, map);
//	clock_t end = clock();
//	printf("time taken %f\n", (end - start) / (float)CLOCKS_PER_SEC);
//
//	crf_plane.del_mem_higherorder();
//
//	// save the output
//	labelToRGB(map, dataset->testImageFiles[files]);
//
//	del_meminit();
//	delete[] map;
//
//
//	string output;
//
//	if (outOpt->count == 0) {
//		output = string(dataset->testFolder) + "denseho.txt";
//	}
//	else {
//		output = *outOpt->filename;
//	}
//
//	evaluateGroundTruth(dataset, dataset->testImageFiles, output);
//
//	cout << endl << endl << "finished with processing"
//		<< endl << endl << "Results are stored in '" << output << "'" << endl;
//
//	delete dataset;
//}
//
//static bool ReadProbability(float4** prob, int width, int height, int num_class,
//	Image<DiscretePdf> &img) {
//
//	// each pixel stores a probability distribution
//	img.alloc(width, height);
//	for (int y = 0; y < height; ++y)
//		for (int x = 0; x < width; ++x)
//			img(x, y).resize(num_class);
//
//	for (int y = 0, i = 0; y < height; ++y) {
//		for (int x = 0; x < width; ++x, ++i) {
//			img(x, y)[0] = prob[y][x].x;
//			img(x, y)[1] = prob[y][x].y;
//			img(x, y)[2] = prob[y][x].z;
//			img(x, y)[3] = prob[y][x].w;
//		}
//	}
//
//	for (int y = 0; y < height; ++y)
//		for (int x = 0; x < width; ++x) {
//			float s = img(x, y).sum();
//			if (s > 0.0f && s < 0.9f) {
//				std::cout << "Pdf not normalized: " << s << std::endl;
//			}
//			img(x, y).normalize();
//		}
//
//	return true;
//}
