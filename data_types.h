#pragma once

#include "math_util.h"

#include <map>

#include <ceres/rotation.h>

////////////////////////////////////////////////////////////////////////////////

#define FLOAT_MAX             (1.f * INT_MAX)
#define FLOAT_MIN             (1.f * INT_MIN)

typedef std::pair<uint, uint>  pii;
typedef std::map<uint, uint>   mii;

template <typename T>
struct Image {
  uint2 size;
  std::vector<T> vdata;

  Image() {
    size  = make_uint2(0);
  }

  Image(const uint2 & s) {
    size  = make_uint2(0);
    alloc(s);
  }

  Image(const uint& cols, const uint& rows) {
    size  = make_uint2(0);
    alloc(cols, rows);
    SetZero();
  }

  void alloc(const uint2 & s) {
    if (s.x == size.x && s.y == size.y)
      return;
    vdata.resize(s.x * s.y);
    size  = s;
  }

  void alloc(const uint cols, const uint rows) {
    if (cols == size.x && rows == size.y)
      return;
    vdata.resize(cols * rows);
    size  = make_uint2(cols, rows);
  }

  T& operator()(const uint &c, const uint &r) {
    return vdata[c + size.x * r];
  }

  const T& operator()(const uint &c, const uint &r) const {
    return vdata[c + size.x * r];
  }

  T& operator[](const uint2 & pix) {
    return vdata[pix.x + size.x * pix.y];
  }

  const T& operator[](const uint2 & pix) const {
    return vdata[pix.x + size.x * pix.y];
  }

  Image<T>& operator=(const Image<T> &other) {
    if (vdata.size() == 0) {
      size = other.size;
      vdata.resize(size.x * size.y);
    }
    vdata.assign(other.vdata.begin(), other.vdata.end());
    return *this;
  }

  T* data() {
    return vdata.data();
  }

  const T* data() const {
    return vdata.data();
  }

  void SetZero() {
    memset(vdata.data(), 0, size.x * size.y * sizeof(T));
  }

  void Free() {
    vdata.clear();
    size  = make_uint2(0);
  }
};

typedef Image<float2> ImageFloat2;
typedef Image<float4> ImageFloat4;
typedef Image<uchar3> ImageUchar3;
typedef Image<uchar4> ImageUchar4;
typedef Image<float>  ImageFloat;
typedef Image<uint>   ImageUint;

////////////////////////////////////////////////////////////////////////////////
// Matrix & Pose

struct CLMatrix {

  CLMatrix() {
    data[0] = make_float4(1.f, 0.f, 0.f, 0.f);
    data[1] = make_float4(0.f, 1.f, 0.f, 0.f);
    data[2] = make_float4(0.f, 0.f, 1.f, 0.f);
  }

  CLMatrix(float* w, float* t) {
    float r[9];
    ceres::AngleAxisToRotationMatrix(w, r);

    data[0] = make_float4(r[0], r[3], r[6], t[0]);
    data[1] = make_float4(r[1], r[4], r[7], t[1]);
    data[2] = make_float4(r[2], r[5], r[8], t[2]);
  }

  //////////////////////////////////////////////////////////

  inline float4 Translation() const {
    float4 t;
    t = make_float4(data[0].w, data[1].w, data[2].w);
    return t;
  }

  //////////////////////////////////////////////////////////

  inline void SetTranslation(float4 t) {
    data[0].w = t.x;
    data[1].w = t.y;
    data[2].w = t.z;
  }

  //////////////////////////////////////////////////////////

  inline CLMatrix Rotation() {
    CLMatrix R = *this;
    R.data[0].w = 0.f;
    R.data[1].w = 0.f;
    R.data[2].w = 0.f;
    return R;
  }

  //////////////////////////////////////////////////////////

  float4 AngleAxis() const {
    double r[9], a[3];
    r[0] = data[0].x;
    r[1] = data[1].x;
    r[2] = data[2].x;
    r[3] = data[0].y;
    r[4] = data[1].y;
    r[5] = data[2].y;
    r[6] = data[0].z;
    r[7] = data[1].z;
    r[8] = data[2].z;
    ceres::RotationMatrixToAngleAxis(r, a);

    return make_float4((float) a[0], (float) a[1], (float) a[2], 0.f);
  }
  //////////////////////////////////////////////////////////

  inline float4 Rotate(const float4 &x) const {
    return make_float4(dot(data[0], x),
                       dot(data[1], x),
                       dot(data[2], x));
  }

  //////////////////////////////////////////////////////////

  inline CLMatrix Rotate(const CLMatrix &B) {
    CLMatrix R;

    float4 c1 = make_float4(B.data[0].x, B.data[1].x, B.data[2].x);
    float4 c2 = make_float4(B.data[0].y, B.data[1].y, B.data[2].y);
    float4 c3 = make_float4(B.data[0].z, B.data[1].z, B.data[2].z);

    for (int i = 0; i < 3; ++i)
      R.data[i] = make_float4 (dot(data[i], c1),
                               dot(data[i], c2),
                               dot(data[i], c3));
    return R;
  }

  //////////////////////////////////////////////////////////

  inline float4 operator*(const float4 &v) const {
    // FIXME: must consider v.w. 
    return make_float4(dot(data[0], v) + data[0].w,
                       dot(data[1], v) + data[1].w,
                       dot(data[2], v) + data[2].w);
  }

  //////////////////////////////////////////////////////////

  inline CLMatrix operator*(const CLMatrix &B) {
    CLMatrix T;

    float4 c1 = make_float4(B.data[0].x, B.data[1].x, B.data[2].x);
    float4 c2 = make_float4(B.data[0].y, B.data[1].y, B.data[2].y);
    float4 c3 = make_float4(B.data[0].z, B.data[1].z, B.data[2].z);
    float4 c4 = make_float4(B.data[0].w, B.data[1].w, B.data[2].w);

    for (int i = 0; i < 3; ++i)
      T.data[i] = make_float4(dot(data[i], c1),
                                dot(data[i], c2),
                                dot(data[i], c3),
                                dot(data[i], c4) + data[i].w);
    return T;
  }

  //////////////////////////////////////////////////////////

  inline CLMatrix Inverse() const {
    CLMatrix  Ti;
    Ti.data[0] = make_float4(data[0].x, data[1].x, data[2].x);
    Ti.data[1] = make_float4(data[0].y, data[1].y, data[2].y);
    Ti.data[2] = make_float4(data[0].z, data[1].z, data[2].z);

    float4 ti = Ti * Translation();

    Ti.data[0].w = - ti.x;
    Ti.data[1].w = - ti.y;
    Ti.data[2].w = - ti.z;

    return Ti;
  }

  //////////////////////////////////////////////////////////

  inline bool IsNaN() const {
      return std::isnan(data[0].x) || std::isnan(data[0].y) || 
	     std::isnan(data[0].z) || std::isnan(data[0].w) ||
             std::isnan(data[1].x) || std::isnan(data[1].y) || 
	     std::isnan(data[1].z) || std::isnan(data[1].w) ||
             std::isnan(data[2].x) || std::isnan(data[2].y) || 
	     std::isnan(data[2].z) || std::isnan(data[2].w);
  }

  inline void getMatrix(float *m) const {
      m[0] = data[0].x, m[4] = data[0].y, m[8] = data[0].z,  m[12] = data[0].w;
      m[1] = data[1].x, m[5] = data[1].y, m[9] = data[1].z,  m[13] = data[1].w;
      m[2] = data[2].x, m[6] = data[2].y, m[10] = data[2].z, m[14] = data[2].w;
      m[3] = 0.0f,      m[7] = 0.0f,      m[11] = 0.0f,      m[15] = 1.0f;
  }

  inline void setMatrix(float *m) {   // OpenGL style column major
      data[0].x = m[0];  data[0].y = m[4]; data[0].z = m[8];  data[0].w = m[12];
      data[1].x = m[1];  data[1].y = m[5]; data[1].z = m[9];  data[1].w = m[13];
      data[2].x = m[2];  data[2].y = m[6]; data[2].z = m[10]; data[2].w = m[14];
  }

  float4 data[3];
};

////////////////////////////////////////////////////////////////////////////////

inline std::ostream& operator<<(std::ostream& os, const CLMatrix& T) {
  float4 d;
  d = T.data[0];
  os << d.x << " " << d.y << " " << d.z << " " << d.w << std::endl;
  d = T.data[1];
  os << d.x << " " << d.y << " " << d.z << " " << d.w << std::endl;
  d = T.data[2];
  os << d.x << " " << d.y << " " << d.z << " " << d.w << std::endl;
  return os;
}

////////////////////////////////////////////////////////////////////////////////

typedef CLMatrix CLPose;

////////////////////////////////////////////////////////////////////////////////

