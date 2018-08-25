#pragma once

//#include "../data_types.h"

#include <vector>
using namespace std;

/**
 * Discrete probability distribution function given by an array
 */
class DiscretePdf {
public:
    DiscretePdf();
	DiscretePdf(const vector<float> &arr);
	
    float probability(int k) const {
        return pdf[k];
    }

    float operator[](int k) const {
        return pdf[k];
    }

    float& operator[](int k) {
        return pdf[k];
    }

    float* data() {
        return pdf.data();
    }

    size_t size() const { 
        return pdf.size(); 
    }

    void reserve(size_t capacity) { 
        pdf.reserve(capacity); 
    }

    void resize(size_t size) { 
        pdf.resize(size); 
        //memset(pdf.data(), 0, sizeof(float) * pdf.size());
    }
    
    void setUniform() {
        if (pdf.size() == 0) return;
        float v = 1.0f / pdf.size();
        for (int i = 0; i < pdf.size(); ++i) pdf[i] = v;
    }

    void setZero() {
        for (int i = 0; i < pdf.size(); ++i) {
            pdf[i] = 0.0f;
        }
        is_zero = true;
    }
    
    DiscretePdf& operator*(const DiscretePdf &p);
    DiscretePdf& operator+(const DiscretePdf &p);

    float sum() const {
        float s = 0.0f;
        for (int i = 0; i < pdf.size(); ++i) {
            s += pdf[i];
        }
        return s;
    }

    void normalize();

protected:
	vector<float> pdf;

    bool is_zero;
};
