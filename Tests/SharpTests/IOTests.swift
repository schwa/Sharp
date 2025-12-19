import Foundation
import simd
@testable import Sharp
import Testing

@Suite("Image Loader Tests")
struct ImageLoaderTests {
    @Test("Image resize maintains aspect info")
    func imageResize() {
        let pixels = [Float](repeating: 0.5, count: 4 * 4 * 3)
        let image = LoadedImage(
            pixels: pixels,
            width: 4,
            height: 4,
            focalLengthPx: 100
        )

        let resized = resizeImage(image, to: 8, targetHeight: 8)

        #expect(resized.width == 8)
        #expect(resized.height == 8)
        #expect(resized.pixels.count == 8 * 8 * 3)
        #expect(resized.focalLengthPx == 200)
    }

    @Test("Image vertical flip")
    func imageFlip() {
        let pixels: [Float] = [
            1, 0, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1
        ]
        let image = LoadedImage(pixels: pixels, width: 2, height: 2, focalLengthPx: 100)

        let flipped = flipImageVertically(image)

        #expect(flipped.pixels[0] == 0)
        #expect(flipped.pixels[1] == 0)
        #expect(flipped.pixels[2] == 1)
    }

    @Test("Load non-existent image throws")
    func loadNonExistent() {
        let url = URL(fileURLWithPath: "/nonexistent/path/image.jpg")
        #expect(throws: ImageLoadError.self) {
            try loadImage(from: url)
        }
    }
}

@Suite("PLY Writer Tests")
struct PLYWriterTests {
    @Test("Save empty Gaussians throws")
    func saveEmptyGaussians() throws {
        let gaussians = Gaussians3D()
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test_empty.ply")

        #expect(throws: PLYWriteError.self) {
            try savePLY(gaussians, focalLengthPx: 500, imageWidth: 640, imageHeight: 480, to: url)
        }
    }

    @Test("Save creates valid PLY file")
    func saveCreatesFile() throws {
        let gaussians = Gaussians3D(
            meanVectors: [SIMD3<Float>(0, 0, 1)],
            singularValues: [SIMD3<Float>(0.1, 0.1, 0.1)],
            quaternions: [SIMD4<Float>(1, 0, 0, 0)],
            colors: [SIMD3<Float>(0.5, 0.5, 0.5)],
            opacities: [0.8]
        )

        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test_save.ply")
        try? FileManager.default.removeItem(at: url)

        try savePLY(gaussians, focalLengthPx: 500, imageWidth: 640, imageHeight: 480, to: url)

        #expect(FileManager.default.fileExists(atPath: url.path))

        let data = try Data(contentsOf: url)
        #expect(!data.isEmpty)

        let header = String(data: data.prefix(100), encoding: .ascii) ?? ""
        #expect(header.hasPrefix("ply"))
        #expect(header.contains("element vertex 1"))

        try? FileManager.default.removeItem(at: url)
    }
}
