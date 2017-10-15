#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "proposal.h"
#include "types.h"
#include "types.hpp"
#include "utils.h"
#include "vis_wrapper.h"
#include "geometry.hpp"
#include "segmentation.hpp" // felzenswalb segmenter
#include "measures.h"
#include "eigen_extensions.h"

using namespace std;

// Segmentation parameters
constexpr float threshold = 0.3f;
constexpr int min_segment_size = 500;
constexpr int max_segment_size = 1000000;

constexpr int min_proposal_size = 750; // minimum size of a region to be considered as a valid proposal
constexpr float min_dimension = 0.05f;
constexpr float max_dimension = 2.5f;

constexpr float thin_threshold = 0.05;
constexpr float flat_threshold = 0.001;

// Objectness parameters
constexpr float symmetry_cloud_normals_tradeoff = 0.2f;
constexpr float local_convexity_radius = 0.0075f;
constexpr float smoothness_radius = 0.01;
constexpr float smoothness_nbins = 8;

constexpr bool enable_visualization = false;


int propose_regions(
    const std::vector<float4>& vertex_vec,
    const std::vector<float4>& normal_vec,
    const std::set<pii>& eset,
    std::vector<region>& regions)
{
    float dim_ratio;
    VisWrapper v;
    v.vis_.setBackgroundColor(0.5, 0.5, 0.5);
    v.vis_.setCameraPosition(0, 0, 0, 0.3, 0.3, 0.3, 0, -1, 0);

    cloud_type::Ptr cloud{new cloud_type};
    normal_cloud_type::Ptr normals{new normal_cloud_type};
    cloud_type::Ptr display{new cloud_type};

    cloud->points.resize(vertex_vec.size());
    normals->points.resize(normal_vec.size());
    for (int i = 0; i < vertex_vec.size(); ++i) {
        cloud->points[i].x = vertex_vec[i].x;
        cloud->points[i].y = vertex_vec[i].y;
        cloud->points[i].z = vertex_vec[i].z;
        normals->points[i].normal_x = normal_vec[i].x;
        normals->points[i].normal_y = normal_vec[i].y;
        normals->points[i].normal_z = normal_vec[i].z;
    }


    printf("Constructing edge graph based on mesh connectivity...\n");
    vector< pair<int,int> > edge_pairs;
    int N = cloud->points.size();
    int E = eset.size();
    segmentation_edge* edges = new segmentation_edge[E];
    auto it = eset.begin();
    for(int i = 0; i < E; ++i, ++it) {
        edges[i].a = it->first;
        edges[i].b = it->second;

        normal_type& n1 = normals->points[edges[i].a];
        normal_type& n2 = normals->points[edges[i].b];
        point_type& p1 = cloud->points[edges[i].a];
        point_type& p2 = cloud->points[edges[i].b];

        // calculate tangent vector
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float dz = p2.z - p1.z;
        float dd = sqrt(dx * dx + dy * dy + dz * dz);
        dx /= dd; dy /= dd; dz /=dd;
        float dot = n1.normal_x * n2.normal_x + n1.normal_y * n2.normal_y + n1.normal_z * n2.normal_z;
        float dot2 = n2.normal_x * dx + n2.normal_y * dy + n2.normal_z * dz;
        float ww = 1.0 - dot;
        if (dot2 > 0) ww = ww * ww;
        edges[i].w = ww;
    }


    printf("Segmenting scene with threshold...\n");
    dsforest* u = segment_graph(N, E, edges, threshold);
    // post-process segmentation by joining small segments
    for(int i = 0; i < E; ++i) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_segment_size) || (u->size(b) < min_segment_size))) {
            u->merge(a, b);
        }
    }
    delete[] edges;

    printf("u->n: %d\n", u->nsets());

    if (enable_visualization) {
        printf("Displaying segmentation...\n");
        map<int, rgb> color_map;
        for (int i = 0; i < N; ++i) {
            point_type pt;
            rgb color;
            int s = u->find(i);
            if (color_map.count(s) > 0) {
                color = color_map[s];
            } else {
                color.r = (rand() % 205) + 50;
                color.g = (rand() % 205) + 50;
                color.b = (rand() % 205) + 50;
                color_map[s] = color;
            }
            pt.x = cloud->points[i].x;
            pt.y = cloud->points[i].y;
            pt.z = cloud->points[i].z;
            pt.r = color.r;
            pt.g = color.g;
            pt.b = color.b;
            display->push_back(pt);
        }
        v.showCloud(display);
        while (true) {
            char key = v.waitKey();
            if (key == 's') pcl::io::savePLYFile("segment.ply", *display);
            if (key == 'q') break;
        }
    }


    vector<cloud_type::Ptr> segments;
    vector<normal_cloud_type::Ptr> segment_normals;
    vector< set<int> > segment_points;
    map<int, int> segment_map;
    for(int i = 0; i < N; ++i) {
        int s = u->find(i);
        int sz = u->size(s);
        if (sz < min_proposal_size) {
            continue;
        }
        if(segment_map.count(s) > 0) {
            int id = segment_map[s];
            cloud_type::Ptr& segment = segments[id];
            segment->push_back(cloud->points[i]);
            segment->height++;
            normal_cloud_type::Ptr& segment_normal = segment_normals[id];
            segment_normal->push_back(normals->points[i]);
            segment_normal->height++;
            segment_points[id].insert(i);
        } else {
            cloud_type::Ptr segment{new cloud_type};
            segment->height = 1;
            segment->width = 1;
            segment->push_back(cloud->points[i]);
            segments.push_back(segment);
            normal_cloud_type::Ptr segment_normal{new normal_cloud_type};
            segment_normal->height = 1;
            segment_normal->width = 1;
            segment_normal->push_back(normals->points[i]);
            segment_normals.push_back(segment_normal);
            set<int> points;
            points.insert(i);
            segment_points.push_back(points);
            segment_map[s] = segments.size() - 1;
        }
    }
    int R = segments.size(); // number of regions
    printf("Found %d components using k=%f!\n", R, threshold);

    vector<point_type> segment_centroids(R);
    vector<normal_type> normal_centroids(R);
    for (int i = 0; i < R; ++i) {
        pcl::CentroidPoint<point_type> centroid_point;
        pcl::CentroidPoint<normal_type> centroid_normal;
        for (int j = 0; j < segments[i]->points.size(); ++j) {
            centroid_point.add(segments[i]->points[j]);
            centroid_normal.add(segment_normals[i]->points[j]);
        }
        centroid_point.get(segment_centroids[i]);
        centroid_normal.get(normal_centroids[i]);
    }


    printf("Projecting all segments to their eigenbasis...\n");
    regions.resize(R);
    for(int i = 0; i < R; ++i) {
        regions[i].type = region::unknown;
        float lambda1, lambda2;
        eigen_basis_transform(segments[i], segment_normals[i],
                              segments[i], segment_normals[i],
                              lambda1, lambda2);
        Eigen::MatrixXf pts = segments[i]->getMatrixXfMap();
        Eigen::VectorXf dims = pts.rowwise().maxCoeff() - pts.rowwise().minCoeff();
        if ((lambda2 < flat_threshold) && (dims(0) > max_dimension || dims(1) > max_dimension || dims(2) > max_dimension)) {
            regions[i].type = region::structure;
            continue;
        }
    }

    printf("Computing shape objectness measures on all segments...\n");
    Eigen::MatrixXf measures(R, 4);
    Eigen::MatrixXf rows(R, 4);
    measures.setZero();
    rows.setZero();
    int NP = 0;
    for(int s = 0; s < R; ++s) {
        if (s % 5 == 0)
            printf("Processing segment %d/%d\n", s, R);
        if (regions[s].type == region::structure) continue;
        if (segments[s]->points.size() > max_segment_size) continue;
        measures(s, 0) = scoreCompactness(segments[s]);
        measures(s, 1) = scoreSymmetry(segments[s], segment_normals[s], symmetry_cloud_normals_tradeoff);
        measures(s, 2) = scoreLocalConvexity(segments[s], segment_normals[s], local_convexity_radius);
        measures(s, 3) = scoreSmoothness(segments[s], segment_normals[s], smoothness_radius, smoothness_nbins);
        rows.row(NP++) = measures.row(s);
    }

    // Compute mean and standard deviation along each column
    Eigen::MatrixXf mm = rows.block(0, 0, NP, 4);
    Eigen::VectorXf mmean = mm.colwise().sum() / mm.rows();
    mm.rowwise() -= mmean.transpose();
    Eigen::MatrixXf mstd = (mm.colwise().squaredNorm() / mm.rows()).cwiseSqrt();

    Eigen::MatrixXf measures_normalized(measures);
    measures_normalized.rowwise() -= mmean.transpose();
    measures_normalized = measures_normalized.cwiseQuotient(mstd.replicate(measures_normalized.rows(), 1));
    Eigen::VectorXf objectness = measures_normalized.rowwise().sum() / measures_normalized.cols(); // objectness is just the average

    vector<fipair> candidates;
    for (int i = 0; i < R; ++i) {
        if (regions[i].type == region::structure) continue;
        candidates.push_back(fipair{objectness(i), i});
    }
    std::sort(candidates.begin(), candidates.end(), fiComparatorDescend);

    int max_proposal = std::min(20, std::max(10, (int)candidates.size() / 5));
    for (int i = 0; i < max_proposal; ++i) {
        regions[candidates[i].second].type = region::object;
    }

    printf("R: %d, NP: %d, best: %d\n", R, NP, max_proposal);

    for (int i = 0; i < regions.size(); ++i) {
        regions[i].indices.resize(segment_points[i].size());
        int k = 0;
        for (auto it = segment_points[i].begin(); it != segment_points[i].end(); ++it) {
            regions[i].indices[k++] = *it;
        }
    }

    if (enable_visualization) {
        printf("Displaying structure proposal and seeds...\n");
        v.holdOff();
        dim_ratio = 0.2;
        for(int i = 0; i < display->points.size(); ++i) {
            display->points[i].r = 128;
            display->points[i].g = 128;
            display->points[i].b = 128;
        }

        for (int s = 0; s < regions.size(); ++s) {
            if (regions[s].type == region::object) {
                for (auto it = regions[s].indices.begin(); it != regions[s].indices.end(); ++it) {
                    int i = *it;
                    display->points[i].r = 163;
                    display->points[i].g = 190;
                    display->points[i].b = 140;
                }
            } else if (regions[s].type == region::structure) {
                for (auto it = regions[s].indices.begin(); it != regions[s].indices.end(); ++it) {
                    int i = *it;
                    display->points[i].r = 191;
                    display->points[i].g = 97;
                    display->points[i].b = 106;
                }
            }
        }
        v.showCloud(display);
        while (true) {
            char key = v.waitKey();
            if (key == 's') pcl::io::savePLYFile("structure.ply", *display);
            if (key == 'q') break;
        }
    }

    printf("Merging..\n");

    // Region merging code goes here
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<region> merged_regions;
        vector<point_type> merged_centroids;
        for (int i = 0; i < regions.size(); ++i) {
            if (regions[i].type != region::unknown) {
                merged_regions.push_back(regions[i]);
                merged_centroids.push_back(segment_centroids[i]);
                continue;
            }
            bool merged = false;
            for (int j = 0; j < merged_regions.size(); ++j) {
                if (merged_regions[j].type != region::object) continue;
                float dx = segment_centroids[i].x - merged_centroids[j].x;
                float dy = segment_centroids[i].y - merged_centroids[j].y;
                float dz = segment_centroids[i].z - merged_centroids[j].z;
                float dist = sqrt(dx * dx + dy * dy + dz * dz);
                if (dist < 0.3f) {
                    Vector4f p1 = merged_centroids[j].getVector4fMap() * merged_regions[j].indices.size();
                    Vector4f p2 = segment_centroids[i].getVector4fMap() * regions[i].indices.size();
                    Vector4f p = (p1 + p2) / (merged_regions[j].indices.size() + regions[i].indices.size());
                    merged_centroids[j].x = p[0];
                    merged_centroids[j].y = p[1];
                    merged_centroids[j].z = p[2];
                    merged_regions[j].indices.insert(merged_regions[j].indices.end(), regions[i].indices.begin(), regions[i].indices.end());
                    changed = true;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                merged_regions.push_back(regions[i]);
                merged_centroids.push_back(segment_centroids[i]);
            }
        }
        regions = merged_regions;
        segment_centroids = merged_centroids;
        printf("%d %d\n", regions.size(), segment_centroids.size());
    }

    if (enable_visualization) {
        printf("Displaying object proposals...\n");
        v.holdOff();
        dim_ratio = 0.2;
        for(int i = 0; i < display->points.size(); ++i) {
            display->points[i].r = 128;
            display->points[i].g = 128;
            display->points[i].b = 128;
        }

        for (int s = 0; s < regions.size(); ++s) {
            if (regions[s].type == region::object) {
                for (auto it = regions[s].indices.begin(); it != regions[s].indices.end(); ++it) {
                    int i = *it;
                    display->points[i].r = 163;
                    display->points[i].g = 190;
                    display->points[i].b = 140;
                }
            } else if (regions[s].type == region::structure) {
                for (auto it = regions[s].indices.begin(); it != regions[s].indices.end(); ++it) {
                    int i = *it;
                    display->points[i].r = 191;
                    display->points[i].g = 97;
                    display->points[i].b = 106;
                }
            }
        }
        v.showCloud(display);
        while (true) {
            char key = v.waitKey();
            if (key == 's') pcl::io::savePLYFile("final.ply", *display);
            if (key == 'q') break;
        }
    }


    for (int i = 0; i < regions.size(); ++i) {
        if (regions[i].type == region::structure)
            regions[i].score = 1.0f;
        else
            regions[i].score = objectness(i);
    }
}


int propose_regions_fast(
    const std::vector<float4>& vertex_vec,
    const std::vector<float4>& normal_vec,
    const std::set<pii>& eset,
    std::vector<region>& regions)
{
    cloud_type::Ptr cloud{new cloud_type};
    normal_cloud_type::Ptr normals{new normal_cloud_type};
    cloud_type::Ptr display{new cloud_type};

    cloud->points.resize(vertex_vec.size());
    normals->points.resize(normal_vec.size());
    for (int i = 0; i < vertex_vec.size(); ++i) {
        cloud->points[i].x = vertex_vec[i].x;
        cloud->points[i].y = vertex_vec[i].y;
        cloud->points[i].z = vertex_vec[i].z;
        normals->points[i].normal_x = normal_vec[i].x;
        normals->points[i].normal_y = normal_vec[i].y;
        normals->points[i].normal_z = normal_vec[i].z;
    }


    printf("Constructing edge graph based on mesh connectivity...\n");
    vector< pair<int,int> > edge_pairs;
    int N = cloud->points.size();
    int E = eset.size();
    segmentation_edge* edges = new segmentation_edge[E];
    auto it = eset.begin();
    for(int i = 0; i < E; ++i, ++it) {
        edges[i].a = it->first;
        edges[i].b = it->second;

        normal_type& n1 = normals->points[edges[i].a];
        normal_type& n2 = normals->points[edges[i].b];
        point_type& p1 = cloud->points[edges[i].a];
        point_type& p2 = cloud->points[edges[i].b];

        // calculate tangent vector
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float dz = p2.z - p1.z;
        float dd = sqrt(dx * dx + dy * dy + dz * dz);
        dx /= dd; dy /= dd; dz /=dd;
        float dot = n1.normal_x * n2.normal_x + n1.normal_y * n2.normal_y + n1.normal_z * n2.normal_z;
        float dot2 = n2.normal_x * dx + n2.normal_y * dy + n2.normal_z * dz;
        float ww = 1.0 - dot;
        if (dot2 > 0) ww = ww * ww;
        edges[i].w = ww;
    }


    printf("Segmenting scene with threshold...\n");
    dsforest* u = segment_graph(N, E, edges, threshold);
    // post-process segmentation by joining small segments
    for(int i = 0; i < E; ++i) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_segment_size) || (u->size(b) < min_segment_size))) {
            u->merge(a, b);
        }
    }
    delete[] edges;

    printf("u->n: %d\n", u->nsets());

    vector<cloud_type::Ptr> segments;
    vector<normal_cloud_type::Ptr> segment_normals;
    vector< set<int> > segment_points;
    map<int, int> segment_map;
    for(int i = 0; i < N; ++i) {
        int s = u->find(i);
        int sz = u->size(s);
        if (sz < min_proposal_size) {
            continue;
        }
        if(segment_map.count(s) > 0) {
            int id = segment_map[s];
            cloud_type::Ptr& segment = segments[id];
            segment->push_back(cloud->points[i]);
            segment->height++;
            normal_cloud_type::Ptr& segment_normal = segment_normals[id];
            segment_normal->push_back(normals->points[i]);
            segment_normal->height++;
            segment_points[id].insert(i);
        } else {
            cloud_type::Ptr segment{new cloud_type};
            segment->height = 1;
            segment->width = 1;
            segment->push_back(cloud->points[i]);
            segments.push_back(segment);
            normal_cloud_type::Ptr segment_normal{new normal_cloud_type};
            segment_normal->height = 1;
            segment_normal->width = 1;
            segment_normal->push_back(normals->points[i]);
            segment_normals.push_back(segment_normal);
            set<int> points;
            points.insert(i);
            segment_points.push_back(points);
            segment_map[s] = segments.size() - 1;
        }
    }
    int R = segments.size(); // number of regions
    printf("Found %d components using k=%f!\n", R, threshold);
    regions.resize(R);
    for(int i = 0; i < R; ++i) {
        regions[i].type = region::unknown;
        regions[i].indices.insert(regions[i].indices.end(),
                                  segment_points[i].begin(),
                                  segment_points[i].end());
    }
    return R;
}
