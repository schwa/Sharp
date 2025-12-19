import simd
@testable import Sharp
import Testing

@Suite("Linear Algebra Tests")
struct LinearAlgebraTests {
    @Test("Identity quaternion produces identity rotation")
    func identityQuaternion() {
        let q = SIMD4<Float>(1, 0, 0, 0)
        let rotation = rotationMatrixFromQuaternion(q)

        #expect(abs(rotation[0][0] - 1) < 1e-5)
        #expect(abs(rotation[1][1] - 1) < 1e-5)
        #expect(abs(rotation[2][2] - 1) < 1e-5)
        #expect(abs(rotation[0][1]) < 1e-5)
        #expect(abs(rotation[0][2]) < 1e-5)
    }

    @Test("Quaternion to rotation matrix roundtrip")
    func quaternionRotationRoundtrip() {
        let angle: Float = .pi / 2
        let q = SIMD4<Float>(cos(angle / 2), 0, 0, sin(angle / 2))

        let rotation = rotationMatrixFromQuaternion(q)
        let qBack = quaternionFromRotationMatrix(rotation)

        let dot = abs(simd_dot(q, qBack))
        #expect(abs(dot - 1) < 1e-4, "Quaternion roundtrip failed")
    }

    @Test("Rotation matrix determinant is 1")
    func rotationMatrixDeterminant() {
        let testQuaternions: [SIMD4<Float>] = [
            SIMD4(1, 0, 0, 0),
            normalize(SIMD4(1, 1, 0, 0)),
            normalize(SIMD4(1, 0, 1, 0)),
            normalize(SIMD4(1, 0, 0, 1)),
            normalize(SIMD4(1, 1, 1, 1))
        ]

        for q in testQuaternions {
            let rotation = rotationMatrixFromQuaternion(q)
            let det = rotation.determinant
            #expect(abs(det - 1) < 1e-4, "Rotation determinant should be 1")
        }
    }

    @Test("Covariance matrix compose/decompose roundtrip")
    func covarianceRoundtrip() {
        let originalQuat = normalize(SIMD4<Float>(1, 0.5, 0.3, 0.1))
        let originalScales = SIMD3<Float>(1.0, 0.5, 0.3)

        let covariance = composeCovarianceMatrix(quaternion: originalQuat, singularValues: originalScales)
        let (recoveredQuat, recoveredScales) = decomposeCovarianceMatrix(covariance)

        let sortedOriginal = [originalScales.x, originalScales.y, originalScales.z].sorted(by: >)
        let sortedRecovered = [recoveredScales.x, recoveredScales.y, recoveredScales.z].sorted(by: >)

        for i in 0..<3 {
            #expect(abs(sortedOriginal[i] - sortedRecovered[i]) < 0.01,
                    "Scale mismatch at index \(i)")
        }

        let recomposed = composeCovarianceMatrix(quaternion: recoveredQuat, singularValues: recoveredScales)

        for col in 0..<3 {
            for row in 0..<3 {
                #expect(abs(covariance[col][row] - recomposed[col][row]) < 0.01,
                        "Covariance mismatch at [\(col)][\(row)]")
            }
        }
    }
}
