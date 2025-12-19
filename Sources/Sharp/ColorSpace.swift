#if !arch(x86_64)
import Foundation
import simd

public enum ColorSpace: UInt8, Sendable, Codable {
    case linearRGB = 0
    case sRGB = 1
}

func linearRGBToSRGB(_ linear: Float) -> Float {
    if linear <= 0.0031308 {
        return linear * 12.92
    }
    return 1.055 * pow(linear, 1.0 / 2.4) - 0.055
}

func linearRGBToSRGB(_ color: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(
        linearRGBToSRGB(color.x),
        linearRGBToSRGB(color.y),
        linearRGBToSRGB(color.z)
    )
}

let sphericalHarmonicsDegree0Coefficient: Float = sqrt(1.0 / (4.0 * .pi))

func rgbToSphericalHarmonics(_ rgb: SIMD3<Float>) -> SIMD3<Float> {
    (rgb - 0.5) / sphericalHarmonicsDegree0Coefficient
}
#endif
