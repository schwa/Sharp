#if !arch(x86_64)
import Foundation
import simd

struct CameraIntrinsics: Sendable {
    var matrix: simd_float4x4

    var fx: Float {
        get { matrix[0][0] }
        set { matrix[0][0] = newValue }
    }

    var fy: Float {
        get { matrix[1][1] }
        set { matrix[1][1] = newValue }
    }

    var cx: Float {
        get { matrix[2][0] }
        set { matrix[2][0] = newValue }
    }

    var cy: Float {
        get { matrix[2][1] }
        set { matrix[2][1] = newValue }
    }

    init() {
        self.matrix = matrix_identity_float4x4
    }

    init(focalLengthPx: Float, imageWidth: Int, imageHeight: Int) {
        self.matrix = simd_float4x4(
            SIMD4(focalLengthPx, 0, 0, 0),
            SIMD4(0, focalLengthPx, 0, 0),
            SIMD4(Float(imageWidth) / 2.0, Float(imageHeight) / 2.0, 1, 0),
            SIMD4(0, 0, 0, 1)
        )
    }

    init(fx: Float, fy: Float, cx: Float, cy: Float) {
        self.matrix = simd_float4x4(
            SIMD4(fx, 0, 0, 0),
            SIMD4(0, fy, 0, 0),
            SIMD4(cx, cy, 1, 0),
            SIMD4(0, 0, 0, 1)
        )
    }

    func scaled(toWidth newWidth: Int, toHeight newHeight: Int, fromWidth: Int, fromHeight: Int) -> Self {
        let scaleX = Float(newWidth) / Float(fromWidth)
        let scaleY = Float(newHeight) / Float(fromHeight)

        return Self(
            fx: fx * scaleX,
            fy: fy * scaleY,
            cx: cx * scaleX,
            cy: cy * scaleY
        )
    }
}

struct CameraExtrinsics: Sendable {
    var matrix: simd_float4x4

    init() {
        self.matrix = matrix_identity_float4x4
    }

    var cameraPosition: SIMD3<Float> {
        let rotation = simd_float3x3(
            SIMD3(matrix[0].x, matrix[0].y, matrix[0].z),
            SIMD3(matrix[1].x, matrix[1].y, matrix[1].z),
            SIMD3(matrix[2].x, matrix[2].y, matrix[2].z)
        )
        let translation = SIMD3(matrix[3].x, matrix[3].y, matrix[3].z)
        return -rotation.transpose * translation
    }
}

func computeUnprojectionMatrix(
    extrinsics: CameraExtrinsics,
    intrinsics: CameraIntrinsics,
    imageWidth: Int,
    imageHeight: Int
) -> simd_float4x4 {
    let ndcMatrix = simd_float4x4(
        SIMD4(2.0 / Float(imageWidth), 0, 0, 0),
        SIMD4(0, 2.0 / Float(imageHeight), 0, 0),
        SIMD4(-1, -1, 1, 0),
        SIMD4(0, 0, 0, 1)
    )

    let projectionMatrix = ndcMatrix * intrinsics.matrix * extrinsics.matrix
    return projectionMatrix.inverse
}
#endif
