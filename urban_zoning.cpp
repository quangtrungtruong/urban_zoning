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

////////////////////////////////////////////////////////////////////////////////

UrbanObject::~UrbanObject() {
}

void UrbanObject::Init() {
	srand(2015);		// seed random generator
	param.clear();

	LoadSatelliteGeotagged();

	/*
	// turn into absolute path to avoid confusion
	if (data_dir[0] != '/' && data_dir[1] != ':') {
	std::ostringstream oss;
	oss << boost::filesystem::current_path().string();
	oss << "/" << data_dir;
	data_dir = oss.str();
	}*/

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
/*void UrbanObject::Replay(string type, bool has_color)
{
	file_correct = data_dir + dataset + "_correct.ply";
	string xml_correct = data_dir + dataset + "_correct.xml";

	LoadPLYWithLabel(data_dir + dataset + "_" + type + ".ply");
	LoadCorres();

	if (has_color) {
		std::cout << "Load vertex color..." << std::endl;
		LoadVertexColor();
	}

	for (int i = 0; i < kfs_num; ++i)
		kfs_corres.push_back(all_corres[kfs_id[i]]);

	// make a graphcut to support easy split without MRF
	GraphSegmentation(mesh_vertex_threshold2, seg_min_size, false,
		graphcut_color_map, graphcut_color_vec, graphcut_label_vec);

	LoadAnnotation(xml_correct);

	GLWindow();
	CvUI();
	AnnotateWindow();


	// Replay all actions
	std::ifstream actions;
	actions.open(data_dir + "actions.txt");

	std::cout << label_vec.size() << std::endl;
	while (!actions.eof()) {
		std::string time;
		std::getline(actions, time);
		std::cout << time << std::endl;

		std::string action_type;
		actions >> action_type;
		std::cout << action_type << std::endl;

		if (anno_window)
			anno_window->pushState();

		if (action_type == "resegment") {
			undo_color.Insert(color_vec);
			undo_label.Insert(label_vec);
			anno_window->pushState();

			uint root_vertex, tail_vertex;
			actions >> root_vertex >> tail_vertex;
			std::cout << "reseg: " << root_vertex << " " << tail_vertex << "\n";
			ReSegment(root_vertex, tail_vertex);
		}

		if (action_type == "graphcut") {
			undo_color.Insert(color_vec);
			undo_label.Insert(label_vec);
			anno_window->pushState();

			std::vector<uint> vertex_id;
			uint size;

			actions >> size;
			vertex_id.resize(size);
			for (int i = 0; i < size; ++i) actions >> vertex_id[i];
			RetrieveGraphcut(vertex_id);
		}

		if (action_type == "merge") {
			undo_color.Insert(color_vec);
			undo_label.Insert(label_vec);
			anno_window->pushState();

			std::vector<uint> lvec;
			uint size, root_vertex;
			actions >> size;
			lvec.resize(size);
			for (int i = 0; i < size; ++i) actions >> lvec[i];
			actions >> root_vertex;
			MergeLabels(lvec, root_vertex);
		}

		if (action_type == "undo") {
			if (undo_color.Size() == 0) {
				cout << "Undo stack is empty" << endl;
			}
			else {
				color_vec = undo_color.Pop();
				label_vec = undo_label.Pop();
				anno_window->popState();
				cv_prev_kf = -1;
			}
		}

		std::string tmp;
		std::getline(actions, tmp);
	}
}
*/


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
	//Load classification score
	/*LoadSatellite(data_dir + dataset + "_prob.ply");
	LoadColorMapXML("nyu_color.xml", globalColormap);
	label_vec.resize(vertex_num);
	nyu_class_vec.resize(vertex_num);*/

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

	//WritePLYWithProbability(data_dir + dataset + "_prob_quant.ply", true, true, true);


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
	 float* gaussian = new float[N * 3];
	 const float sx = 0.1;
	 const float sy = 0.1;
	 const float sz = 0.1;
	 for (int i = 0; i < N; ++i) {
	     gaussian[i * 3 + 0] = pixel_vec[i].x / sx;
	     gaussian[i * 3 + 1] = pixel_vec[i].y / sy;
	     gaussian[i * 3 + 2] = pixel_vec[i].z / sz;
	 }

	// float* surface = new float[N * 6];
	// const float snx = 0.1;
	// const float sny = 0.1;
	// const float snz = 0.1;
	// for (int i = 0; i < N; ++i) {
	//     surface[i * 6 + 0] = vertex_vec[i].x / sx;
	//     surface[i * 6 + 1] = vertex_vec[i].y / sy;
	//     surface[i * 6 + 2] = vertex_vec[i].z / sz;
	//     surface[i * 6 + 3] = normal_vec[i].x / snx;
	//     surface[i * 6 + 4] = normal_vec[i].y / sny;
	//     surface[i * 6 + 5] = normal_vec[i].z / snz;
	// }

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

	 //for (int i = 0; i < N; ++i) {
	 //    label_vec[i] = map[i] + 1;
	 //    nyu_class_vec[i] = label_vec[i];
	 //}

	 //// Recolor
	 //for (int i = 0; i < N; ++i) {
	 //    if (globalColormap.find(label_vec[i]) == globalColormap.end()) {
	 //        std::cout << "Label out of range: " << label_vec[i] << std::endl;
	 //        color_vec[i] = make_uchar4(0, 0, 0);
	 //        continue;
	 //    }
	 //    color_vec[i] = globalColormap[label_vec[i]].color;
	 //}

	// WritePLYWithLabel(data_dir + dataset + "_crf.ply");

	// save XML
	/*PrepareLabelMap();
	Annotation anno;
	vector<RegionAnno> regions;
	this->GetRegionDataForAnnotation(regions);
	anno.importRegions(regions);
	anno.export_to_xml(data_dir + "_crf.xml");*/

	// crf.clearMemory();
	// delete[] unary;
	// delete[] gaussian;
	// delete[] surface;
	// delete[] map;
}

void UrbanObject::ComputeProbability() {
	//LoadPLYWithLabel(data_dir + dataset + ".ply");

	std::vector<std::string> prob_list;
	//GetFileNames(data_dir + "pred/", &prob_list);

	/*std::vector<int> frames;
	for (int i = 0; i < prob_list.size(); ++i)
		frames.push_back(ParseFrame(prob_list[i]));
	std::cout << "Frames with probability: " << frames.size() << std::endl;*/

	/*int probWidth = 480;
	int probHeight = 360;
	int numClass = 37;*/
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

	// Perform a projection on vertices that fall into the same pixel but not captured by all_corres[i]
	// we do a soft depth test to avoid noise artifacts and have smoother texture.
	/*ImageFloat depth_buffer;
	Image<vector<uint> > index_buffer;
	depth_buffer.alloc(cols, rows);
	index_buffer.alloc(cols, rows);
	const float DEPTH_THRESHOLD = 1e-3f;
	const float NO_DEPTH = 1000.0f;*/
	//for (int i = 0; i < frames.size(); ++i) {
	//	int k = frames[i];
	//	std::cout << "Processing frame " << k << std::endl;

	//	Image<DiscretePdf> imgProb;
	//	std::vector<DiscretePdf> vecProb;
	//	/*if (!ReadProbability(prob_list[i], probWidth, probHeight, numClass, imgProb)) {
	//		std::cout << "Error reading file " << prob_list[i] << std::endl;
	//		continue;
	//	}*/
	//	ReadProbability(satellite_prob, probWidth, probHeight, numClass, imgProb);

	//	// projection and depth test to get visible vertices
	//	for (uint y = 0; y < rows; ++y) {
	//		for (uint x = 0; x < cols; ++x) {
	//			depth_buffer(x, y) = NO_DEPTH;
	//			index_buffer(x, y).clear();
	//		}
	//	}

	//	//for (int v = 0; v < vertex_num; ++v) {
	//	//	float4 p = vertex_vec[v];
	//	//	float4 c = all_pose[k].Inverse() * p;
	//	//	if (c.z <= 0) continue;

	//	//	int hx = (int)(cam_K.x * c.x / c.z + cam_K.z);
	//	//	int hy = (int)(cam_K.y * c.y / c.z + cam_K.w);
	//	//	if (hx < 0 || hx >= cols || hy < 0 || hy >= rows) continue;

	//	//	if (fabs(c.z - depth_buffer(hx, hy)) < DEPTH_THRESHOLD) {

	//	//		index_buffer(hx, hy).push_back(v);
	//	//	}

	//	//	if (c.z < depth_buffer(hx, hy) - DEPTH_THRESHOLD) {
	//	//		depth_buffer(hx, hy) = c.z;
	//	//		vector<uint> tmp = index_buffer(hx, hy);

	//	//		index_buffer(hx, hy).clear();
	//	//		index_buffer(hx, hy).push_back(v);

	//	//		for (auto vid : tmp) {		// keep those within the new range
	//	//			float4 p2 = vertex_vec[vid];
	//	//			float4 c2 = all_pose[k].Inverse() * p2;
	//	//			if (fabs(c2.z - depth_buffer(hx, hy)) < DEPTH_THRESHOLD) {
	//	//				index_buffer(hx, hy).push_back(vid);
	//	//			}
	//	//		}

	//	//	}
	//	//}

	//	// update per-vertex probability
	//	for (uint y = 0; y < rows; ++y) {
	//		for (uint x = 0; x < cols; ++x) {
	//			if (index_buffer(x, y).size() == 0) continue;

	//			for (auto vid : index_buffer(x, y)) {
	//				// take corresponding vertex in the lower resolution probability map
	//				int xx = (int)((float)x * probWidth / cols);
	//				int yy = (int)((float)y * probHeight / rows);

	//				/*
	//				if (pdf_vec[vid].isValid())
	//				pdf_vec[vid] = pdf_vec[vid] * imgProb(xx, yy);
	//				else
	//				pdf_vec[vid] = imgProb(xx, yy);
	//				*/

	//				//pdf_vec[vid] = pdf_vec[vid] + imgProb(xx, yy);
	//				// int argmax_pdf = 0, argmax_img = 0;
	//				// float max_prob_pdf = 0.0f, max_prob_img = 0;
	//				// for (int k = 0; k < numClass; ++k) {
	//				//     if (pdf_vec[vid][k] > max_prob_pdf) {
	//				//         max_prob_pdf = pdf_vec[vid][k];
	//				//         argmax_pdf = k;
	//				//     }
	//				//     if (imgProb(xx, yy)[k] > max_prob_img) {
	//				//         max_prob_img = imgProb(xx, yy)[k];
	//				//         argmax_img = k;
	//				//     }
	//				// }
	//				// if (argmax_pdf != argmax_img) confidence[vid]--;
	//				// else confidence[vid]++;
	//				// pdf_vec[vid] = pdf_vec[vid] * imgProb(xx, yy);

	//				int argmax_img = 0;
	//				float max_prob_img = 0.0f;
	//				for (int k = 0; k < numClass; ++k) {
	//					if (imgProb(xx, yy)[k] > max_prob_img) {
	//						max_prob_img = imgProb(xx, yy)[k];
	//						argmax_img = k;
	//					}
	//				}
	//				pdf_vec[vid][argmax_img] += 1.0;
	//			}
	//		}
	//	}
	//}

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

	//WritePLYWithProbability(data_dir + dataset + "_prob.ply", true, true, true);
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