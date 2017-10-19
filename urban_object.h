#pragma once

#include "crf/pdf.h"
#include "math_util.h"
#include <string>
#include <map>

#define VERTEX_SHIFT      1
#define MERGE_THR         1000
#define COV_EPSILON       0.001

using namespace std;

template <class T>
struct UndoStack {
	void Insert(T v) {
		if (undo_vec.size() == 100)
			undo_vec.erase(undo_vec.begin());		// FIXME: this can cause inconsistency with annotation undo stack
		undo_vec.push_back(v);
	}

	T Pop() {
		T v = undo_vec.back();
		undo_vec.pop_back();
		return v;
	}

	size_t Size() {
		return undo_vec.size();
	}

	void Free() {
		undo_vec.clear();
	}

	vector<T> undo_vec;
};

	//////////////////////////////////////////////////////////////////////////////

	/*inline void AscendingOrder(const uint v0, const uint v1, uint* a, uint* b) {
		if (v0 < v1) {
			*a = v0; *b = v1;
		}
		else {
			*a = v1; *b = v0;
		}
	}*/

	//////////////////////////////////////////////////////////////////////////////

	/*inline void AddNeighbors(uint v0, uint v1, uint v2) {
		ngb_vec[v0].insert(v1);
		ngb_vec[v0].insert(v2);
		ngb_vec[v1].insert(v0);
		ngb_vec[v1].insert(v2);
		ngb_vec[v2].insert(v0);
		ngb_vec[v2].insert(v1);
	}*/

	//////////////////////////////////////////////////////////////////////////////

	/*void Init();
	void ParseVolume(const int4    voxels,
		const short2* volume,
		const bool       set_face = NULL);*/

	/*void LoadPLY(string filename);
	void LoadPLYWithLabel(string filename);
	bool WritePLY(string filename, bool save_normal = true, bool save_color = true, bool save_face = true);
	bool WritePLYWithLabel(string filename, bool save_normal = true, bool save_color = true, bool save_face = true);
	void WriteLabelPLY(string filename, const set<uint> &label, bool binary = true, uchar4 color = make_uchar4(0, 0, 0, 0));*/

class UrbanObject {
public:
	UrbanObject(const char*, const char*);
	virtual ~UrbanObject();

	void GenerateVirtualViews();
	void GenerateDisperancyViews();
	void ShowMesh();
	void GraphcutAndMRF(bool has_vertex_color = false);
	void RunDenseCRF(bool ho_enabled, bool cooc_enabled);
	void ComputeProbability();
	bool AskForSave();
	
	void ComputeStats(string type);

	int               cv_curr_kf;
	int               cv_prev_kf;
	bool              cv_mouse_click;
	bool              cv_ctrl_click;
	bool              cv_shift_click;
	double            cv_factor;
	string            cv_win_name;
	int               cv_tab_max;
	uint UNLABEL;

	int2   xy8[8];

	uint   pixel_num;
	int   rows;
	int   cols;
	int num_geotaggeds;
	float4** satellite_prob;
	int **satellite_zone;
	float4* geotagged_prob;
	float4** nearest_geotagged_dis;
	int4 *geotagged_info;
	unsigned char *im_orig;
	vector<uint>        label_vec;

	string city;
	string data_dir;
	map<string, string> param;

	bool  use_kf_pose;

	int   row_4;
	int   col_4;
	int   max_window_height;
	int   max_window_width;
	int   interval;
	int   kfs_num;
	int   all_num;
	int   vcg_mask;
	int   vnum_per_face;
	int   edges[256];
	int   faces[256][16];
	int   seg_min_size;
	int   seg_min_size2;
	int	count_resegment;
	int	count_merge;
	int	count_graphcut;
	int num_class = 4;

	float iso_value;
	float METER_PER_VOXEL;
	float VOXEL_PER_METER;
	float track_threshold;
	float rsme_threshold;
	float mesh_color_threshold;
	float mesh_vertex_threshold;
	float mesh_vertex_threshold2;
	float vertex_threshold;
	float mrf_rgb_weight;
	float mrf_potential;
	float mrf_iteration;
	float rgbd_c_threshold;

	uchar4 unlabel_color;

	bool debug_correspondence;
	vector<float4> debug_vertices;
	vector<float4> debug_lines;
	double cur_eigen[3][3];
	double example_eigen[3][3];
	float4 cur_centroid;
	float4 example_centroid;

	std::vector<DiscretePdf> pdf_vec;

public:
	enum DisplayMode {                    // presentation modes
		PROBABILITY,
		SEGMENTATION,
		NYU_CLASS,
		SEGMENTATION_GRAPHCUT,
		SEGMENTATION_MRF,
		TEXTURE,
		GREY,
		CLOUD,
		AABB,
		OBB,
		IMAGE_BASED,
		REFLECTANCE,
		SHADING,
		NUM_DISPLAY_MODES
	};

protected:
	DisplayMode display_mode;
	bool capturing_screenshots;
	void collectAllScreenshots();

private:
	void Init();
	void LoadSatelliteGeotagged();
};