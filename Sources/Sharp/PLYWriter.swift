#if !arch(x86_64)
import Foundation
import simd

enum PLYWriteError: Error, LocalizedError {
    case fileCreationFailed(URL)
    case writeFailed(String)
    case invalidData(String)

    var errorDescription: String? {
        switch self {
        case .fileCreationFailed(let url):
            return "Failed to create PLY file: \(url.path)"
        case .writeFailed(let message):
            return "PLY write error: \(message)"
        case .invalidData(let message):
            return "Invalid data: \(message)"
        }
    }
}

/// Save Gaussians to PLY file format
public func savePLY(
    _ gaussians: Gaussians3D,
    focalLengthPx: Float,
    imageWidth: Int,
    imageHeight: Int,
    to url: URL,
    colorSpace: ColorSpace = .linearRGB
) throws {
    guard !gaussians.isEmpty else {
        throw PLYWriteError.invalidData("No Gaussians to save")
    }

    var output = Data()

    let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(gaussians.count)
        property float x
        property float y
        property float z
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        element extrinsic 16
        property float extrinsic
        element intrinsic 9
        property float intrinsic
        element image_size 2
        property uint image_size
        element frame 2
        property int frame
        element disparity 2
        property float disparity
        element color_space 1
        property uchar color_space
        element version 3
        property uchar version
        end_header

        """
    output.append(header.data(using: .ascii)!)

    for i in 0..<gaussians.count {
        let mean = gaussians.meanVectors[i]
        let scales = gaussians.singularValues[i]
        let quat = gaussians.quaternions[i]
        var color = gaussians.colors[i]
        let opacity = gaussians.opacities[i]

        if colorSpace == .linearRGB {
            color = linearRGBToSRGB(color)
        }

        let sh = rgbToSphericalHarmonics(color)
        let opacityLogit = inverseSigmoid(clamp(opacity, min: 1e-6, max: 1 - 1e-6))

        let scaleLogits = SIMD3<Float>(
            log(max(scales.x, 1e-8)),
            log(max(scales.y, 1e-8)),
            log(max(scales.z, 1e-8))
        )

        appendFloat(&output, mean.x)
        appendFloat(&output, mean.y)
        appendFloat(&output, mean.z)
        appendFloat(&output, sh.x)
        appendFloat(&output, sh.y)
        appendFloat(&output, sh.z)
        appendFloat(&output, opacityLogit)
        appendFloat(&output, scaleLogits.x)
        appendFloat(&output, scaleLogits.y)
        appendFloat(&output, scaleLogits.z)
        appendFloat(&output, quat.x)
        appendFloat(&output, quat.y)
        appendFloat(&output, quat.z)
        appendFloat(&output, quat.w)
    }

    let identity = matrix_identity_float4x4
    for col in 0..<4 {
        for row in 0..<4 {
            appendFloat(&output, identity[col][row])
        }
    }

    let intrinsic: [Float] = [
        focalLengthPx, 0, Float(imageWidth) / 2.0,
        0, focalLengthPx, Float(imageHeight) / 2.0,
        0, 0, 1
    ]
    for value in intrinsic {
        appendFloat(&output, value)
    }

    appendUInt32(&output, UInt32(imageWidth))
    appendUInt32(&output, UInt32(imageHeight))

    appendInt32(&output, 1)
    appendInt32(&output, Int32(gaussians.count))

    let disparities = gaussians.meanVectors.map { 1.0 / $0.z }
    let sortedDisparities = disparities.sorted()
    let p10 = sortedDisparities[max(0, Int(Float(sortedDisparities.count) * 0.1))]
    let p90 = sortedDisparities[min(sortedDisparities.count - 1, Int(Float(sortedDisparities.count) * 0.9))]
    appendFloat(&output, p10)
    appendFloat(&output, p90)

    output.append(UInt8(ColorSpace.sRGB.rawValue))

    output.append(UInt8(1))
    output.append(UInt8(5))
    output.append(UInt8(0))

    do {
        try output.write(to: url)
    } catch {
        throw PLYWriteError.writeFailed(error.localizedDescription)
    }
}

private func appendFloat(_ data: inout Data, _ value: Float) {
    var v = value
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
}

private func appendUInt32(_ data: inout Data, _ value: UInt32) {
    var v = value
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
}

private func appendInt32(_ data: inout Data, _ value: Int32) {
    var v = value
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
}
#endif
