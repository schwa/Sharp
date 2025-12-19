import simd
@testable import Sharp
import Testing

@Suite("Color Space Tests")
struct ColorSpaceTests {
    @Test("Linear RGB to sRGB conversion")
    func linearToSrgb() {
        let linear = SIMD3<Float>(0.214, 0.214, 0.214)
        let srgb = linearRGBToSRGB(linear)

        #expect(abs(srgb.x - 0.5) < 0.01)
        #expect(abs(srgb.y - 0.5) < 0.01)
        #expect(abs(srgb.z - 0.5) < 0.01)
    }

    @Test("Linear RGB to sRGB scalar conversion")
    func linearToSrgbScalar() {
        #expect(abs(linearRGBToSRGB(Float(0.0)) - 0.0) < 1e-5)
        #expect(abs(linearRGBToSRGB(Float(1.0)) - 1.0) < 1e-5)

        #expect(abs(linearRGBToSRGB(Float(0.214)) - 0.5) < 0.01)
    }

    @Test("RGB to spherical harmonics")
    func rgbToSH() {
        let rgb = SIMD3<Float>(0.3, 0.5, 0.7)
        let sh = rgbToSphericalHarmonics(rgb)

        #expect(sh.x != 0 || sh.y != 0 || sh.z != 0)
    }
}
