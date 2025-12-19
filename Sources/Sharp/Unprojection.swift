#if !arch(x86_64)
import Foundation
import simd

func unprojectGaussians(
    _ gaussians: Gaussians3D,
    extrinsics: CameraExtrinsics,
    intrinsics: CameraIntrinsics,
    imageWidth: Int,
    imageHeight: Int
) -> Gaussians3D {
    let unprojectionMatrix = computeUnprojectionMatrix(
        extrinsics: extrinsics,
        intrinsics: intrinsics,
        imageWidth: imageWidth,
        imageHeight: imageHeight
    )

    let affineTransform = simd_float3x4(
        SIMD4(unprojectionMatrix[0].x, unprojectionMatrix[0].y, unprojectionMatrix[0].z, unprojectionMatrix[0].w),
        SIMD4(unprojectionMatrix[1].x, unprojectionMatrix[1].y, unprojectionMatrix[1].z, unprojectionMatrix[1].w),
        SIMD4(unprojectionMatrix[2].x, unprojectionMatrix[2].y, unprojectionMatrix[2].z, unprojectionMatrix[2].w)
    )

    return applyAffineTransform(gaussians, transform: affineTransform)
}

func applyAffineTransform(_ gaussians: Gaussians3D, transform: simd_float3x4) -> Gaussians3D {
    let linear = simd_float3x3(
        SIMD3(transform[0].x, transform[0].y, transform[0].z),
        SIMD3(transform[1].x, transform[1].y, transform[1].z),
        SIMD3(transform[2].x, transform[2].y, transform[2].z)
    )
    let offset = SIMD3<Float>(transform[0].w, transform[1].w, transform[2].w)

    let newMeans = gaussians.meanVectors.map { mean in
        linear * mean + offset
    }

    var newQuaternions: [SIMD4<Float>] = []
    var newSingularValues: [SIMD3<Float>] = []

    newQuaternions.reserveCapacity(gaussians.count)
    newSingularValues.reserveCapacity(gaussians.count)

    for i in 0..<gaussians.count {
        let oldCovariance = composeCovarianceMatrix(
            quaternion: gaussians.quaternions[i],
            singularValues: gaussians.singularValues[i]
        )

        let newCovariance = linear * oldCovariance * linear.transpose
        let (newQuat, newScales) = decomposeCovarianceMatrix(newCovariance)

        newQuaternions.append(newQuat)
        newSingularValues.append(newScales)
    }

    return Gaussians3D(
        meanVectors: newMeans,
        singularValues: newSingularValues,
        quaternions: newQuaternions,
        colors: gaussians.colors,
        opacities: gaussians.opacities
    )
}

func unprojectGaussians(
    _ gaussians: Gaussians3D,
    intrinsics: CameraIntrinsics,
    imageWidth: Int,
    imageHeight: Int
) -> Gaussians3D {
    unprojectGaussians(
        gaussians,
        extrinsics: CameraExtrinsics(),
        intrinsics: intrinsics,
        imageWidth: imageWidth,
        imageHeight: imageHeight
    )
}
#endif
