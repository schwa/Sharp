#include <metal_stdlib>
using namespace metal;

float4 quaternionMultiply(float4 a, float4 b) {
    return float4(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

float3x3 quaternionToMatrix(float4 q) {
    q = normalize(q);
    float x = q.y, y = q.z, z = q.w, w = q.x;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    return float3x3(
        float3(1.0 - (yy + zz), xy + wz, xz - wy),
        float3(xy - wz, 1.0 - (xx + zz), yz + wx),
        float3(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

float4 matrixToQuaternion(float3x3 m) {
    float trace = m[0][0] + m[1][1] + m[2][2];
    float4 q;

    if (trace > 0) {
        float s = sqrt(trace + 1.0) * 2.0;
        q.x = 0.25 * s;
        q.y = (m[1][2] - m[2][1]) / s;
        q.z = (m[2][0] - m[0][2]) / s;
        q.w = (m[0][1] - m[1][0]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q.x = (m[1][2] - m[2][1]) / s;
        q.y = 0.25 * s;
        q.z = (m[1][0] + m[0][1]) / s;
        q.w = (m[2][0] + m[0][2]) / s;
    } else if (m[1][1] > m[2][2]) {
        float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q.x = (m[2][0] - m[0][2]) / s;
        q.y = (m[1][0] + m[0][1]) / s;
        q.z = 0.25 * s;
        q.w = (m[2][1] + m[1][2]) / s;
    } else {
        float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q.x = (m[0][1] - m[1][0]) / s;
        q.y = (m[2][0] + m[0][2]) / s;
        q.z = (m[2][1] + m[1][2]) / s;
        q.w = 0.25 * s;
    }

    return normalize(q);
}

kernel void extractAndUnproject(
    device const half* meanData [[buffer(0)]],
    device const half* scaleData [[buffer(1)]],
    device const half* quatData [[buffer(2)]],
    device const half* colorData [[buffer(3)]],
    device const half* opacityData [[buffer(4)]],
    device float3* outMeans [[buffer(5)]],
    device float3* outScales [[buffer(6)]],
    device float4* outQuats [[buffer(7)]],
    device float3* outColors [[buffer(8)]],
    device float* outOpacities [[buffer(9)]],
    constant uint& meanStride [[buffer(10)]],
    constant uint& scaleStride [[buffer(11)]],
    constant uint& quatStride [[buffer(12)]],
    constant uint& colorStride [[buffer(13)]],
    constant uint& opacityStride [[buffer(14)]],
    uint id [[thread_position_in_grid]]
) {
    uint meanBase = id * meanStride;
    outMeans[id] = float3(float(meanData[meanBase]), float(meanData[meanBase + 1]), float(meanData[meanBase + 2]));

    uint scaleBase = id * scaleStride;
    outScales[id] = float3(float(scaleData[scaleBase]), float(scaleData[scaleBase + 1]), float(scaleData[scaleBase + 2]));

    uint quatBase = id * quatStride;
    outQuats[id] = float4(float(quatData[quatBase]), float(quatData[quatBase + 1]), float(quatData[quatBase + 2]), float(quatData[quatBase + 3]));

    uint colorBase = id * colorStride;
    outColors[id] = float3(float(colorData[colorBase]), float(colorData[colorBase + 1]), float(colorData[colorBase + 2]));

    outOpacities[id] = float(opacityData[id * opacityStride]);
}
