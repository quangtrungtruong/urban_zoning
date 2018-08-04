#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "proposal.h"
#include <iostream>

using namespace std;

void propose_regions_fast(int4 *geotagged_info, int size, int height, int width, int w, std::vector<region>& regions)
{
    regions.resize(size);
    for(int i = 0; i < size; ++i) {
		regions[i].indices.clear();
        regions[i].type = geotagged_info[i].z;
		int2 bottom, top;
		bottom.x = geotagged_info[i].x - w / 2 > 0 ? geotagged_info[i].x - w / 2 : 0;
		bottom.y = geotagged_info[i].y - w / 2 > 0 ? geotagged_info[i].y - w / 2 : 0;
		top.x = geotagged_info[i].x + w / 2 < height ? geotagged_info[i].x + w / 2 : height-1;
		top.y = geotagged_info[i].y + w / 2 < width ? geotagged_info[i].y + w / 2 : width-1;

		for (int k = bottom.x; k < top.x; k++)
			for (int l = bottom.y; l < top.y;l++)
				regions[i].indices.insert(regions[i].indices.end(), k*width + l);
    }
    return;
}