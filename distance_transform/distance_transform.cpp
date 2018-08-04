#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include <iostream>
#include <fstream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "pnmfile.h"
#include "imconv.h"
#include "dt.h"

using namespace std;
using namespace cv;

//Copyright (C) 2006 Pedro Felzenszwalb
void distance_transform(string input_name, string output_name){
	string output_name_pgm = output_name + ".pgm";
	string output_name_txt = output_name + ".txt";
	// compute dt
	Mat mat_in = imread(input_name, CV_LOAD_IMAGE_GRAYSCALE);
	image<float> *out = new image<float>(mat_in.cols, mat_in.rows, false);

	for (int y = 0; y < mat_in.rows; y++) {
		for (int x = 0; x < mat_in.cols; x++) {
			if (mat_in.at<uchar>(y, x) < 122)
				imRef(out, x, y) = 0;
			else
				imRef(out, x, y) = INF;
		}
	}

	dt(out);

	// take square roots
	for (int y = 0; y < out->height(); y++) {
		for (int x = 0; x < out->width(); x++) {
			imRef(out, x, y) = sqrt(imRef(out, x, y));
		}
	}

	// save to distance transform to text file
	ofstream infile(output_name_txt);
	infile << mat_in.rows << " " << mat_in.cols << endl;
	double diagonal_length = sqrt(out->height()*out->height() + out->width()*out->width());
	for (int y = 0; y < mat_in.rows; y++) {
		for (int x = 0; x < mat_in.cols; x++)
			infile << out->access[y][x] / diagonal_length << " ";
		infile << endl;
	}
	infile.close();

	// convert to grayscale
	image<uchar> *gray = imageFLOATtoUCHAR(out);

	// save output
	savePGM(gray, output_name_pgm.c_str());

	//delete input;
	delete out;
	delete gray;
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cout << "Format for parameter: <path> <city>";
		return 0;
	}
	cout << "finish";
	std::string data_dir = argv[1];
	std::string city = argv[2];

	int   pixel_num;
	int   rows;
	int   cols;
	int num_geotaggeds;
	string dir_sat = data_dir + "//satellite//" + city + ".txt";
	string dir_geo = data_dir + "//geotagged//" + city + ".txt";
	string dir_out = data_dir + "//distance_transform//" + city;
	ifstream satellite_fin(dir_sat.c_str());
	ifstream geotagged_fin(dir_geo.c_str());

	if (satellite_fin.is_open())
	{
		satellite_fin >> rows >> cols;
		cout << "row-col: " << rows << " " << cols << endl;
		satellite_fin.close();
	}

	Mat img0(rows, cols, CV_8U, Scalar(255));
	Mat img1(rows, cols, CV_8U, Scalar(255));
	Mat img2(rows, cols, CV_8U, Scalar(255));
	Mat img3(rows, cols, CV_8U, Scalar(255));
	pixel_num = rows*cols;
	std::string delimiter = ",";
	if (geotagged_fin.is_open())
	{
		geotagged_fin >> num_geotaggeds;
		cout << "num-geo " << num_geotaggeds << endl;
		std::string line;
		std::getline(geotagged_fin, line);

		for (int i = 0; i < num_geotaggeds; i++)
		{
			std::getline(geotagged_fin, line);
			size_t pos = 0;
			std::string token;
			int x, y, v;
			double att;

			pos = line.find(delimiter);
			token = line.substr(0, pos);
			x = atoi(token.c_str());
			line.erase(0, pos + delimiter.length());
			pos = line.find(delimiter);
			token = line.substr(0, pos);
			y = atoi(token.c_str());
			line.erase(0, pos + delimiter.length());

			for (int k = 0; k<209; k++) {
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				line.erase(0, pos + delimiter.length());
			}

			pos = line.find(delimiter);
			token = line.substr(0, pos);
			v = atoi(token.c_str());
			line.erase(0, pos + delimiter.length());

			//cout << "(" << i << " " << x << " " << y<< " "  << v<< ")";
			if (v == 0)
				img0.at<uchar>(y, x) = 0;
			else if (v == 1)
				img1.at<uchar>(y, x) = 0;
			else if (v == 2)
				img2.at<uchar>(y, x) = 0;
			else if (v == 3)
				img3.at<uchar>(y, x) = 0;

		}
		satellite_fin.close();
		geotagged_fin.close();
	}

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	compression_params.push_back(0);

	imwrite(dir_out + "_0.pbm", img0, compression_params);
	imwrite(dir_out + "_1.pbm", img1, compression_params);
	imwrite(dir_out + "_2.pbm", img2, compression_params);
	imwrite(dir_out + "_3.pbm", img3, compression_params);

	distance_transform(dir_out+"_0.pbm", dir_out+"_dt_0");
	distance_transform(dir_out+"_1.pbm", dir_out+"_dt_1");
	distance_transform(dir_out+"_2.pbm", dir_out+"_dt_2");
	distance_transform(dir_out+"_3.pbm", dir_out+"_dt_3");

	return 0;
}