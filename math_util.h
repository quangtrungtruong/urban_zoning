#pragma once

#include <cmath>

typedef unsigned char uchar;
typedef unsigned int uint;
typedef struct
{
	int x, y;
} int2;
typedef struct
{
	int x, y, z, w;
} int4;
typedef struct
{
	uint x, y;
} uint2;
typedef struct
{
	uint x, y, z, w;
} uint4;
typedef struct
{
	float x, y;
} float2;
typedef struct
{
    float x, y, z;
} float3;
typedef struct
{
	float x, y, z, w;
} float4;
typedef struct
{
    uchar x, y, z;
} uchar3;
typedef struct
{
	uchar x, y, z, w;
} uchar4;
typedef struct
{
	short x, y;
} short2;

////////////////////////////////////////////////////////////////////////////////

inline uint2 make_uint2(uint x, uint y) {
    uint2 t;
    t.x = x; t.y = y;
    return t;
}

inline uint2 make_uint2(uint i) {
    uint2 t;
    t.x = i; t.y = i;
    return t;
}

inline uint4 make_uint3(uint i) {
  uint4 t;
  t.x = i; t.y = i; t.z = i; t.w = 0;
  return t;
}

inline uint4 make_uint3(uint x, uint y, uint z) {
  uint4 t;
  t.x = x; t.y = y; t.z = z; t.w = 0;
  return t;
}

inline uint4 make_uint3(int4 a) {
  uint4 t;
  t.x = (uint) a.x; t.y = (uint) a.y; t.z = (uint) a.z; t.w = 0;
  return t;
}

inline uint4 make_uint3(float4 a) {
  uint4 t;
  t.x = (uint) a.x; t.y = (uint) a.y; t.z = (uint) a.z; t.w = 0;
  return t;
}

inline uint4 make_uint4(uint x, uint y, uint z) {
  uint4 t;
  t.x = x; t.y = y; t.z = z; t.w = 0;
  return t;
}

inline uint4 make_uint4(uint x, uint y, uint z, uint w) {
  uint4 t;
  t.x = x; t.y = y; t.z = z; t.w = w;
  return t;
}

inline int2 make_int2(int x, int y) {
  int2 t;
  t.x = x; t.y = y;
  return t;
}

inline int4 make_int3(int i) {
  int4 t;
  t.x = i; t.y = i; t.z = i; t.w = 0;
  return t;
}

inline int4 make_int3(int x, int y, int z) {
  int4 t;
  t.x = x; t.y = y; t.z = z; t.w = 0;
  return t;
}

inline int4 make_int3(uint4 a) {
  int4 t;
  t.x = (int) a.x; t.y = (int) a.y; t.z = (int) a.z; t.w = 0;
  return t;
}

inline int4 make_int3(float4 a) {
  int4 t;
  t.x = (int) a.x; t.y = (int) a.y; t.z = (int) a.z; t.w = 0;
  return t;
}

inline int4 make_int4(int x, int y, int z) {
  int4 t;
  t.x = x; t.y = y; t.z = z; t.w = 0;
  return t;
}

inline int4 make_int4(int x, int y, int z, int w) {
  int4 t;
  t.x = x; t.y = y; t.z = z; t.w = w;
  return t;
}

inline uchar4 make_uchar3(uchar x, uchar y, uchar z) {
  uchar4 t;
  t.x = x; t.y = y; t.z = z; t.w = 0;
  return t;
}

inline uchar4 make_uchar4(uchar x, uchar y, uchar z) {
  uchar4 t;
  t.x = x; t.y = y; t.z = z; t.w = 0;
  return t;
}

inline uchar4 make_uchar4(uchar x, uchar y, uchar z, uchar w) {
  uchar4 t;
  t.x = x; t.y = y; t.z = z; t.w = w;
  return t;
}

inline float2 make_float2(float x, float y) {
    float2 t;
    t.x = x; t.y = y;
    return t;
}

inline float2 make_float2(float f) {
    float2 t;
    t.x = f; t.y = f;
    return t;
}

inline float3 make_float3(float x, float y, float z) {
    float3 t;
    t.x = x; t.y = y; t.z = z;
    return t;
}

inline float3 make_float3(float f) {
    float3 t;
    t.x = f; t.y = f; t.z = f;
    return t;
}

inline float3 make_float3(float4 f) {
    float3 t;
    t.x = f.x; t.y = f.y; t.z = f.z;
    return t;
}

inline float4 make_float4(float x, float y, float z, float w = 0.0f) {
  float4 t;
  t.x = x; t.y = y; t.z = z; t.w = w;
  return t;
}

inline float4 make_float4(float3 f, float w = 0.0f) {
  float4 t;
  t.x = f.x; t.y = f.y; t.z = f.z; t.w = w;
  return t;
}

inline float4 make_float4(float f) {
    float4 t;
    t.x = f; t.y = f; t.z = f; t.w = f;
    return t;
}

////////////////////////////////////////////////////////////////////////////////

inline uint2 operator>>(uint2 a, int b) {
  return make_uint2(a.x >> b, a.y >> b);
}

////////////////////////////////////////////////////////////////////////////////

inline int min(int a, int b) {
  return a < b ? a : b;
}

inline uchar min(uchar a, uchar b) {
  return a < b ? a : b;
}

inline float min(float a, float b) {
  return a < b ? a : b;
}

inline int2 min(int2 a, int2 b) {
  return make_int2(min(a.x, b.x), min(a.y, b.y));
}

inline int4 min(int4 a, int4 b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline float3 min(float3 a, float3 b) {
    return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline float4 min(float4 a, float4 b) {
  return make_float4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

////////////////////////////////////////////////////////////////////////////////

inline int max(int a, int b) {
  return a > b ? a : b;
}

inline float max(float a, float b) {
  return a > b ? a : b;
}

inline int2 max(int2 a, int2 b) {
  return make_int2(max(a.x,b.x), max(a.y,b.y));
}

inline int4 max(int4 a, int4 b) {
  return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline float3 max(float3 a, float3 b) {
    return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline float4 max(float4 a, float4 b) {
  return make_float4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

////////////////////////////////////////////////////////////////////////////////

inline int clamp(int f, int a, int b) {
  return max(a, min(f, b));
}

inline float clamp(float f, float a, float b) {
  return max(a, min(f, b));
}

inline float3 clamp(float3 v, float a, float b) {
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline float3 clamp(float3 v, float3 a, float3 b) {
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

inline float4 clamp(float4 v, float a, float b) {
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

////////////////////////////////////////////////////////////////////////////////

inline bool operator<(float4 a, float4 b) {
  return (a.x < b.x && a.y < b.y && a.z < b.z);
}

////////////////////////////////////////////////////////////////////////////////

inline bool operator>(float4 a, float4 b) {
  return (a.x > b.x && a.y > b.y && a.z > b.z);
}

////////////////////////////////////////////////////////////////////////////////

inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float dot(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

////////////////////////////////////////////////////////////////////////////////

inline float3 cross(float3 a, float3 b) {
    float3 n;
    n.x = (a.y * b.z) - (a.z * b.y);
    n.y = (a.z * b.x) - (a.x * b.z);
    n.z = (a.x * b.y) - (a.y * b.x);
    return n;
}

inline float4 cross(float4 a, float4 b) {
  float4 n;
  n.x = (a.y * b.z) - (a.z * b.y);
  n.y = (a.z * b.x) - (a.x * b.z);
  n.z = (a.x * b.y) - (a.y * b.x);
  return n;
}

////////////////////////////////////////////////////////////////////////////////

inline float3 operator-(float3 f) {
    return make_float3(-f.x, -f.y, -f.z);
}

inline float4 operator-(float4 f) {
  return make_float4(-f.x, -f.y, -f.z);
}

////////////////////////////////////////////////////////////////////////////////

inline int2 operator-(int2 a, int2 b) {
  return make_int2(a.x - b.x, a.y - b.y);
}

////////////////////////////////////////////////////////////////////////////////

inline int2 operator+(int2 a, int2 b) {
  return make_int2(a.x + b.x, a.y + b.y);
}

inline int4 operator+(int4 a, int4 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline uint4 operator+(uint4 a, uint4 b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z);
}

////////////////////////////////////////////////////////////////////////////////

inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline int4 operator-(int4 a, int4 b) {
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

////////////////////////////////////////////////////////////////////////////////

inline float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 operator*(float b, float3 a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b);
}

inline float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline uint4 operator*(uint4 a, int b) {
  return make_uint3(a.x * b, a.y * b, a.z * b);
}

////////////////////////////////////////////////////////////////////////////////

inline float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}

inline float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline float4 operator/(float4 a, float b) {
  return make_float4(a.x / b, a.y / b, a.z / b);
}

inline float4 operator/(float4 a, float4 b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z);
}

////////////////////////////////////////////////////////////////////////////////

inline void operator+=(float2 &a, float2 b) {
    a.x += b.x; a.y += b.y;
}

inline void operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline void operator+=(float4 &a, float4 b) {
  a.x += b.x; a.y += b.y; a.z += b.z;
}

inline void operator+=(float3 &a, float b) {
  a.x += b; a.y += b; a.z += b;
}

////////////////////////////////////////////////////////////////////////////////

inline void operator-=(float3 &a, float b) {
    a.x -= b; a.y -= b; a.z -= b;
}

inline void operator-=(float4 &a, float b) {
    a.x -= b; a.y -= b; a.z -= b;
}

////////////////////////////////////////////////////////////////////////////////

inline void operator*=(float3 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline void operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

////////////////////////////////////////////////////////////////////////////////

inline void operator/=(float3 &a, float b) {
    a.x /= b; a.y /= b; a.z /= b;
}

inline void operator/=(float4 &a, float b) {
    a.x /= b; a.y /= b; a.z /= b;
}

////////////////////////////////////////////////////////////////////////////////
//  Common math functions
////////////////////////////////////////////////////////////////////////////////

inline float3 normalize(float3 v) {
    float length = v.x * v.x + v.y * v.y + v.z * v.z;
    float oneOverLength = 1.0f / sqrtf(length);
    v.x *= oneOverLength;
    v.y *= oneOverLength;
    v.z *= oneOverLength;
    return v;
}

inline float4 normalize(float4 v) {
  float length = v.x * v.x + v.y * v.y + v.z * v.z;
  float oneOverLength = 1.0f / sqrtf(length);
  v.x *= oneOverLength;
  v.y *= oneOverLength;
  v.z *= oneOverLength;
  return v;
}

////////////////////////////////////////////////////////////////////////////////

inline float4 fceilf(float4 v) {
  return make_float4(ceil(v.x), ceil(v.y), ceil(v.z));
}

////////////////////////////////////////////////////////////////////////////////

inline float Norm(float3 vec) {
    return sqrtf(dot(vec, vec));
}

inline float Norm(float4 vec) {
  return sqrtf(dot(vec, vec));
}

////////////////////////////////////////////////////////////////////////////////

inline float4 floorf(float4 v) {
  return make_float4(floorf(v.x), floorf(v.y), floorf(v.z));
}

////////////////////////////////////////////////////////////////////////////////

inline float fracf(float v) {
  return v - floorf(v);
}

inline float4 fracf(float4 v) {
  return make_float4(fracf(v.x), fracf(v.y), fracf(v.z));
}

inline bool isnan3(float3 v) {
    return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z);
}

inline bool isnan4(float4 v) {
    return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isnan(v.w);
}

inline bool iszero4(float4 v) {
    return (v.x == 0.0f) && (v.y == 0.0f) && (v.z == 0.0f) && (v.w == 0.0f);
}

////////////////////////////////////////////////////////////////////////////////

