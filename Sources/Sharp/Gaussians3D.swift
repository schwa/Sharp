#if !arch(x86_64)
import Foundation
import simd

/// Represents a collection of 3D Gaussians for Gaussian splatting
public struct Gaussians3D: Sendable {
    /// Mean positions of each Gaussian [N, 3]
    public var meanVectors: [SIMD3<Float>]

    /// Singular values (scales) of each Gaussian [N, 3]
    public var singularValues: [SIMD3<Float>]

    /// Quaternion rotations (w, x, y, z) for each Gaussian [N, 4]
    public var quaternions: [SIMD4<Float>]

    /// RGB colors for each Gaussian [N, 3] in linear RGB
    public var colors: [SIMD3<Float>]

    /// Opacity values for each Gaussian [N]
    public var opacities: [Float]

    /// Number of Gaussians
    public var count: Int {
        meanVectors.count
    }

    public var isEmpty: Bool {
        meanVectors.isEmpty
    }

    /// Create empty Gaussians3D
    public init() {
        self.meanVectors = []
        self.singularValues = []
        self.quaternions = []
        self.colors = []
        self.opacities = []
    }

    /// Create Gaussians3D with specified capacity
    public init(capacity: Int) {
        self.meanVectors = []
        self.singularValues = []
        self.quaternions = []
        self.colors = []
        self.opacities = []
        self.meanVectors.reserveCapacity(capacity)
        self.singularValues.reserveCapacity(capacity)
        self.quaternions.reserveCapacity(capacity)
        self.colors.reserveCapacity(capacity)
        self.opacities.reserveCapacity(capacity)
    }

    /// Create Gaussians3D with pre-allocated arrays
    public init(
        meanVectors: [SIMD3<Float>],
        singularValues: [SIMD3<Float>],
        quaternions: [SIMD4<Float>],
        colors: [SIMD3<Float>],
        opacities: [Float]
    ) {
        precondition(meanVectors.count == singularValues.count)
        precondition(meanVectors.count == quaternions.count)
        precondition(meanVectors.count == colors.count)
        precondition(meanVectors.count == opacities.count)

        self.meanVectors = meanVectors
        self.singularValues = singularValues
        self.quaternions = quaternions
        self.colors = colors
        self.opacities = opacities
    }
}

func rotationMatrixFromQuaternion(_ q: SIMD4<Float>) -> simd_float3x3 {
    let normalized = normalize(q)
    let w = normalized.x
    let x = normalized.y
    let y = normalized.z
    let z = normalized.w

    let xx = x * x, yy = y * y, zz = z * z
    let xy = x * y, xz = x * z, yz = y * z
    let wx = w * x, wy = w * y, wz = w * z

    return simd_float3x3(
        SIMD3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
        SIMD3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
        SIMD3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy))
    )
}

func quaternionFromRotationMatrix(_ m: simd_float3x3) -> SIMD4<Float> {
    let trace = m[0][0] + m[1][1] + m[2][2]

    var w, x, y, z: Float

    if trace > 0 {
        let s = sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[1][2] - m[2][1]) / s
        y = (m[2][0] - m[0][2]) / s
        z = (m[0][1] - m[1][0]) / s
    } else if m[0][0] > m[1][1], m[0][0] > m[2][2] {
        let s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[1][2] - m[2][1]) / s
        x = 0.25 * s
        y = (m[1][0] + m[0][1]) / s
        z = (m[2][0] + m[0][2]) / s
    } else if m[1][1] > m[2][2] {
        let s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[2][0] - m[0][2]) / s
        x = (m[1][0] + m[0][1]) / s
        y = 0.25 * s
        z = (m[2][1] + m[1][2]) / s
    } else {
        let s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[0][1] - m[1][0]) / s
        x = (m[2][0] + m[0][2]) / s
        y = (m[2][1] + m[1][2]) / s
        z = 0.25 * s
    }

    return SIMD4(w, x, y, z)
}
#endif
