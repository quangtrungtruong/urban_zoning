#include "stats.h"

#include <iostream>
#include <ctime>
#include <cmath>
using namespace std;

Stats::Stats() : a(0.5f), one_minus_a(1.0f - a) {
    last_fps = 0.0f;

    last_tic = 0;
    last_toc = 0;

    total_clocks = 0;
    total_frames = 0;
}

long int Stats::tic() {
    last_tic = clock();
    return last_tic;
}

long int Stats::tic(std::string message) {
    cout << "# " << message << endl;
    return tic();
}

long int Stats::toc(bool print) {
    last_toc = clock();

    total_clocks += last_toc - last_tic;
    total_frames++;

    if (print) print_elapsed();

    return last_toc;
}

float Stats::elapsed() const {
    return static_cast<float>(last_toc - last_tic) * 1000 / CLOCKS_PER_SEC;
}

float Stats::fps() {
    if (total_clocks == 0) return last_fps;

    float new_fps = last_fps * a + one_minus_a *
                    (total_frames * CLOCKS_PER_SEC / static_cast<float>(total_clocks));

    last_fps = new_fps;
    return new_fps;
}

float Stats::get_total_elapsed() {
    return static_cast<float>(total_clocks) * 1000 / CLOCKS_PER_SEC;
}

void Stats::reset_total_elapsed() {
    total_clocks = 0;
    total_frames = 0;
}

void Stats::print_elapsed_milliseconds() {
    cout << ">>> " << (int)ceil(this->elapsed()) << " ms" << " <<<" << endl;
}

void Stats::print_elapsed_seconds() {
    cout << ">>> " << (int)ceil(this->elapsed() / 1000) << " s" << " <<<" << endl;
}

void Stats::print_elapsed() {
    int s = (int)(this->elapsed() / 1000);
    float ms = this->elapsed() - s * 1000;
    cout << ">>> " << s << " s  " << ms << " ms" << " <<<" << endl;
}
