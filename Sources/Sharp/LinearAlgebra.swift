import Accelerate
import Foundation
import simd

func decomposeCovarianceMatrix(_ covariance: simd_float3x3) -> (quaternion: SIMD4<Float>, singularValues: SIMD3<Float>) {
    var matrix = [
        covariance[0][0], covariance[0][1], covariance[0][2],
        covariance[1][0], covariance[1][1], covariance[1][2],
        covariance[2][0], covariance[2][1], covariance[2][2]
    ]

    var singularValues = [Float](repeating: 0, count: 3)
    var U = [Float](repeating: 0, count: 9)
    var Vt = [Float](repeating: 0, count: 9)

    var m: Int32 = 3
    var n: Int32 = 3
    var lda: Int32 = 3
    var ldu: Int32 = 3
    var ldvt: Int32 = 3
    var info: Int32 = 0

    var jobu = Int8(UnicodeScalar("A").value)
    var jobvt = Int8(UnicodeScalar("A").value)

    var lwork: Int32 = -1
    var workQuery: Float = 0
    sgesvd_(&jobu, &jobvt, &m, &n, &matrix, &lda, &singularValues, &U, &ldu, &Vt, &ldvt, &workQuery, &lwork, &info)

    lwork = Int32(workQuery)
    var work = [Float](repeating: 0, count: Int(lwork))
    sgesvd_(&jobu, &jobvt, &m, &n, &matrix, &lda, &singularValues, &U, &ldu, &Vt, &ldvt, &work, &lwork, &info)

    let rotationMatrix = simd_float3x3(
        SIMD3(U[0], U[1], U[2]),
        SIMD3(U[3], U[4], U[5]),
        SIMD3(U[6], U[7], U[8])
    )

    var fixedRotation = rotationMatrix
    if rotationMatrix.determinant < 0 {
        fixedRotation[2] = -fixedRotation[2]
    }

    let quaternion = quaternionFromRotationMatrix(fixedRotation)

    let scales = SIMD3<Float>(
        sqrt(max(singularValues[0], 0)),
        sqrt(max(singularValues[1], 0)),
        sqrt(max(singularValues[2], 0))
    )

    return (quaternion, scales)
}

func composeCovarianceMatrix(quaternion: SIMD4<Float>, singularValues: SIMD3<Float>) -> simd_float3x3 {
    let rotation = rotationMatrixFromQuaternion(quaternion)
    let diagonal = simd_float3x3(diagonal: singularValues * singularValues)
    return rotation * diagonal * rotation.transpose
}
