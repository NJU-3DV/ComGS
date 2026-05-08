# pragma once

#include <cuda_runtime.h>

#define PBR_EPS 1e-7f
#define PBR_PI 3.1415926535f
#define PBR_MAX_RAY 512


__forceinline__ __device__ float3 operator+(const float& a, const float3& b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}

__forceinline__ __device__ float3 operator+(const float3& a, const float b) { 
    return make_float3(a.x + b, a.y + b, a.z + b); 
}

__forceinline__ __device__ float3 operator+(const float3& a, const float3& b) { 
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); 
}

__forceinline__ __device__ float3& operator+= (float3& a, const float& b) {
    a = a + b;
    return a;
}

__forceinline__ __device__ float3& operator+= (float3& a, const float3& b) {
    a = a + b;
    return a;
}

__forceinline__ __device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __device__ float3 operator-(const float& a, const float3 b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}

__forceinline__ __device__ float3 operator-(const float3& a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3& operator-= (float3& a, const float& b) {
    a = a - b;
    return a;
}

__forceinline__ __device__ float3& operator-= (float3& a, const float3& b) {
    a = a - b;
    return a;
}

__forceinline__ __device__ float3 operator*(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__forceinline__ __device__ float3 operator*(const float3& a, const float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3& operator*= (float3& a, const float& b) {
    a = a * b;
    return a;
}

__forceinline__ __device__ float3& operator*= (float3& a, const float3& b) {
    a = a * b;
    return a;
}

__forceinline__ __device__ float3 operator/(const float& a, const float3& b) {
    return make_float3(a / b.x, a / b.y, a / b.z);
}

__forceinline__ __device__ float3 operator/(const float3& a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__forceinline__ __device__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__forceinline__ __device__ float3& operator/= (float3& a, const float& b) {
    a = a / b;
    return a;
}

__forceinline__ __device__ float3& operator/= (float3& a, const float3& b) {
    a = a / b;
    return a;
}

__forceinline__ __device__ float3 normalize(const float3& a) {
    float norm = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z) + PBR_EPS;
    return make_float3(a.x / norm, a.y / norm, a.z / norm);
}

__forceinline__ __device__ float dotf(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float geometry_schlick_ggx(const float& n_dot_x, const float& roughness) {
    float r = roughness + 1.0f;
    float k = powf(r, 2) / 8.0f;
    float denom = n_dot_x * (1.0f - k) + k + PBR_EPS;
    return n_dot_x / denom;
}

__forceinline__ __device__ float sum(const float3& a) {
    return a.x + a.y + a.z;
}

__forceinline__ __device__ float3 cross(
    const float3& v1, const float3& v2){
    float3 result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}


__forceinline__ __device__ float3 tangent2world(
    const float3& normal, const float3& vec){
    // from tangent-space vector to world-space sample vector
    float3 up = fabs(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return tangent * vec.x + bitangent * vec.y + normal * vec.z;
}


