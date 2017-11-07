#pragma once
#include <string>
#include <vector>
#include <set>
#include "urban_object.h"


struct region
{
    enum { residential, industrial, commercial, other};

	int type = region::other;
    std::vector<int> indices;
    double score = 0.0;
};

void propose_regions_fast(
	int4 *geotagged_info, int size, int height, int width, int w,
	std::vector<region>& regions);

