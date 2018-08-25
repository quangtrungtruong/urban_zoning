#pragma once

#include "crf/pdf.h"
#include "math_util.h"
#include <string>
#include <map>

using namespace std;

class UrbanObject {
public:
	UrbanObject(const char*, const char*);
	UrbanObject();
	virtual ~UrbanObject();
	void RunDenseCRF(bool ho_enabled, bool cooc_enabled, double anpha, double beta, int w, int iteration, float gaussian_w, float bilateral_w, float param, float weight_term4);
	void ComputeProbability();
	void ConvertNearestZoneTypeVec();
	int2   xy8[8];
	uint   pixel_num;
	int   rows;
	int   cols;
	int num_geotaggeds;
	float4** satellite_prob;
	float4** nearest_distance;
	int **satellite_zone;
	float4* geotagged_prob;
	int4 *geotagged_info;
	vector<float4> pixel_vec;
	int** label_matrix;

	float acc = 0;
	string city;
	string data_dir;
	map<string, string> param;
	int num_class = 4;

	std::vector<DiscretePdf> pdf_vec; 
	void Preprocess(std::vector<DiscretePdf> &prop_vec);
	std::vector<DiscretePdf> nearest_p_vec;

private:
	void Init();
	void LoadSatelliteGeotagged();
};