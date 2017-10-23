#pragma once
#include <string>
#include <vector>
#include <set>
#include "urban_object.h"


struct region
{
    enum { unknown, structure, object};

    int type = region::unknown;
    std::vector<int> indices;
    double score = 0.0;
};

int propose_regions_fast(
	int4 *geotagged_info, int size, int heigh, int width,
	std::vector<region>& regions);

