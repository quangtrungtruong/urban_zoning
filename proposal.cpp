#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "proposal.h"

using namespace std;

// Segmentation parameters
//constexpr float threshold = 0.3f;
//constexpr int min_segment_size = 500;
//constexpr int max_segment_size = 1000000;
//
//constexpr int min_proposal_size = 750; // minimum size of a region to be considered as a valid proposal
//constexpr float min_dimension = 0.05f;
//constexpr float max_dimension = 2.5f;
//
//constexpr float thin_threshold = 0.05;
//constexpr float flat_threshold = 0.001;
//
//// Objectness parameters
//constexpr float symmetry_cloud_normals_tradeoff = 0.2f;
//constexpr float local_convexity_radius = 0.0075f;
//constexpr float smoothness_radius = 0.01;
//constexpr float smoothness_nbins = 8;
//
//constexpr bool enable_visualization = false;

int propose_regions_fast(int4 *geotagged_info, int size, int heigh, int width, std::vector<region>& regions, int w)
{
    regions.resize(size);
    for(int i = 0; i < size; ++i) {
        regions[i].type = region::unknown;
		int2 bottom, top;
		bottom.x = geotagged_info[i].x - w / 2 > 0 ? geotagged_info[i].x - w / 2 : 0;
		bottom.y = geotagged_info[i].y - w / 2 > 0 ? geotagged_info[i].y - w / 2 : 0;
		top.x = geotagged_info[i].x + w / 2 < heigh ? geotagged_info[i].x + w / 2 : heigh-1;
		top.y = geotagged_info[i].y + w / 2 < width ? geotagged_info[i].y + w / 2 : width-1;
		for (int k = bottom.x; k <= top.x; k++)
			for (int l = bottom.y; l <= top.y;l++)
				regions[i].indices.insert(regions[i].indices.end(), k*width + l);
    }
    return size;
}