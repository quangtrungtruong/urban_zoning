#pragma once
#include <string>
#include <vector>
#include "mesh_object.h"


struct region
{
    enum { unknown, structure, object};

    int type = region::unknown;
    std::vector<int> indices;
    double score = 0.0;
};

int propose_regions(
    const std::vector<float4>& vertex_vec,
    const std::vector<float4>& normal_vec,
    const std::set<pii>& eset,
    std::vector<region>& regions);

int propose_regions_fast(
    const std::vector<float4>& vertex_vec,
    const std::vector<float4>& normal_vec,
    const std::set<pii>& eset,
    std::vector<region>& regions);
