import simd
@testable import Sharp
import Testing

@Suite("Gaussians3D Tests")
struct Gaussians3DTests {
    @Test("Empty Gaussians3D")
    func emptyGaussians() {
        let gaussians = Gaussians3D()
        #expect(gaussians.isEmpty)
    }

    @Test("Create Gaussians3D with capacity")
    func gaussiansWithCapacity() {
        var gaussians = Gaussians3D(capacity: 100)
        #expect(gaussians.isEmpty)

        gaussians.meanVectors.append(SIMD3(0, 0, 0))
        gaussians.singularValues.append(SIMD3(1, 1, 1))
        gaussians.quaternions.append(SIMD4(1, 0, 0, 0))
        gaussians.colors.append(SIMD3(0.5, 0.5, 0.5))
        gaussians.opacities.append(1.0)

        #expect(gaussians.count == 1)
    }

    @Test("Create Gaussians3D from arrays")
    func gaussiansFromArrays() {
        let count = 10
        let gaussians = Gaussians3D(
            meanVectors: Array(repeating: SIMD3(0, 0, 1), count: count),
            singularValues: Array(repeating: SIMD3(0.1, 0.1, 0.1), count: count),
            quaternions: Array(repeating: SIMD4(1, 0, 0, 0), count: count),
            colors: Array(repeating: SIMD3(1, 0, 0), count: count),
            opacities: Array(repeating: Float(0.8), count: count)
        )

        #expect(gaussians.count == count)
        #expect(gaussians.meanVectors[0].z == 1)
        #expect(gaussians.colors[0].x == 1)
    }
}

@Suite("Camera Tests")
struct CameraTests {
    @Test("Camera intrinsics from focal length")
    func intrinsicsFromFocal() {
        let intrinsics = CameraIntrinsics(focalLengthPx: 500, imageWidth: 640, imageHeight: 480)

        #expect(intrinsics.fx == 500)
        #expect(intrinsics.fy == 500)
        #expect(intrinsics.cx == 320)
        #expect(intrinsics.cy == 240)
    }

    @Test("Camera intrinsics scaling")
    func intrinsicsScaling() {
        let original = CameraIntrinsics(focalLengthPx: 500, imageWidth: 640, imageHeight: 480)
        let scaled = original.scaled(toWidth: 1_280, toHeight: 960, fromWidth: 640, fromHeight: 480)

        #expect(scaled.fx == 1_000)
        #expect(scaled.fy == 1_000)
        #expect(scaled.cx == 640)
        #expect(scaled.cy == 480)
    }

    @Test("Identity extrinsics")
    func identityExtrinsics() {
        let extrinsics = CameraExtrinsics()
        let position = extrinsics.cameraPosition

        #expect(abs(position.x) < 1e-5)
        #expect(abs(position.y) < 1e-5)
        #expect(abs(position.z) < 1e-5)
    }

    @Test("Unprojection matrix identity case")
    func unprojectionMatrixIdentity() {
        let extrinsics = CameraExtrinsics()
        let intrinsics = CameraIntrinsics(focalLengthPx: 100, imageWidth: 200, imageHeight: 200)

        let unprojection = computeUnprojectionMatrix(
            extrinsics: extrinsics,
            intrinsics: intrinsics,
            imageWidth: 200,
            imageHeight: 200
        )

        #expect(unprojection.determinant != 0)
    }
}
