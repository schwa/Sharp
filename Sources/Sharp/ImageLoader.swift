import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

#if canImport(AppKit)
import AppKit
#endif

#if canImport(UIKit)
import UIKit
#endif

enum ImageLoadError: Error, LocalizedError {
    case fileNotFound(URL)
    case invalidImageData
    case unsupportedFormat
    case cgImageCreationFailed
    case exifReadFailed

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Image file not found: \(url.path)"
        case .invalidImageData:
            return "Invalid or corrupted image data"
        case .unsupportedFormat:
            return "Unsupported image format"
        case .cgImageCreationFailed:
            return "Failed to create CGImage"
        case .exifReadFailed:
            return "Failed to read EXIF metadata"
        }
    }
}

/// Result of loading an image
public struct LoadedImage: Sendable {
    /// Pixel data as RGB floats normalized to [0, 1]
    public let pixels: [Float]
    /// Image width
    public let width: Int
    /// Image height
    public let height: Int
    /// Focal length in pixels (from EXIF or estimated)
    public let focalLengthPx: Float

    public init(pixels: [Float], width: Int, height: Int, focalLengthPx: Float) {
        self.pixels = pixels
        self.width = width
        self.height = height
        self.focalLengthPx = focalLengthPx
    }
}

/// Load an image from a file URL
func loadImage(
    from url: URL,
    defaultFocalLengthMultiplier: Float = 1.2
) throws -> LoadedImage {
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw ImageLoadError.fileNotFound(url)
    }

    guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw ImageLoadError.invalidImageData
    }

    guard let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
        throw ImageLoadError.cgImageCreationFailed
    }

    let width = cgImage.width
    let height = cgImage.height

    let focalLengthPx = extractFocalLength(from: imageSource, imageWidth: width, imageHeight: height)
        ?? (Float(width) * defaultFocalLengthMultiplier)

    let pixels = try extractRGBPixels(from: cgImage)

    return LoadedImage(
        pixels: pixels,
        width: width,
        height: height,
        focalLengthPx: focalLengthPx
    )
}

private func extractFocalLength(from source: CGImageSource, imageWidth: Int, imageHeight: Int) -> Float? {
    guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
        return nil
    }

    if let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] {
        if let focalLength35mm = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber {
            let focalMM = focalLength35mm.floatValue
            let imageDiagonal = sqrtf(Float(imageWidth * imageWidth + imageHeight * imageHeight))
            let filmDiagonal = sqrtf(36.0 * 36.0 + 24.0 * 24.0)
            return focalMM * imageDiagonal / filmDiagonal
        }

        if let focalLengthMM = exif[kCGImagePropertyExifFocalLength] as? NSNumber {
            if let pixelXDimension = exif[kCGImagePropertyExifPixelXDimension] as? NSNumber,
               let _ = exif[kCGImagePropertyExifPixelYDimension] as? NSNumber {
                let sensorWidthMM: Float = 24.0
                return focalLengthMM.floatValue * Float(pixelXDimension.intValue) / sensorWidthMM
            }
        }
    }

    return nil
}

private func extractRGBPixels(from cgImage: CGImage) throws -> [Float] {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel

    var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
        throw ImageLoadError.cgImageCreationFailed
    }

    guard let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        throw ImageLoadError.cgImageCreationFailed
    }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    var floatPixels = [Float]()
    floatPixels.reserveCapacity(width * height * 3)

    for y in 0..<height {
        for x in 0..<width {
            let offset = y * bytesPerRow + x * bytesPerPixel
            floatPixels.append(Float(pixelData[offset]) / 255.0)
            floatPixels.append(Float(pixelData[offset + 1]) / 255.0)
            floatPixels.append(Float(pixelData[offset + 2]) / 255.0)
        }
    }

    return floatPixels
}

func flipImageVertically(_ image: LoadedImage) -> LoadedImage {
    var flippedPixels = [Float](repeating: 0, count: image.pixels.count)
    let rowSize = image.width * 3

    for y in 0..<image.height {
        let srcRow = y * rowSize
        let dstRow = (image.height - 1 - y) * rowSize
        for x in 0..<rowSize {
            flippedPixels[dstRow + x] = image.pixels[srcRow + x]
        }
    }

    return LoadedImage(
        pixels: flippedPixels,
        width: image.width,
        height: image.height,
        focalLengthPx: image.focalLengthPx
    )
}

func flipImageHorizontally(_ image: LoadedImage) -> LoadedImage {
    var flippedPixels = [Float](repeating: 0, count: image.pixels.count)

    for y in 0..<image.height {
        for x in 0..<image.width {
            let srcIdx = (y * image.width + x) * 3
            let dstIdx = (y * image.width + (image.width - 1 - x)) * 3
            flippedPixels[dstIdx] = image.pixels[srcIdx]
            flippedPixels[dstIdx + 1] = image.pixels[srcIdx + 1]
            flippedPixels[dstIdx + 2] = image.pixels[srcIdx + 2]
        }
    }

    return LoadedImage(
        pixels: flippedPixels,
        width: image.width,
        height: image.height,
        focalLengthPx: image.focalLengthPx
    )
}

func resizeImage(_ image: LoadedImage, to targetWidth: Int, targetHeight: Int) -> LoadedImage {
    var resizedPixels = [Float](repeating: 0, count: targetWidth * targetHeight * 3)

    let scaleX = Float(image.width) / Float(targetWidth)
    let scaleY = Float(image.height) / Float(targetHeight)

    for y in 0..<targetHeight {
        for x in 0..<targetWidth {
            let srcX = (Float(x) + 0.5) * scaleX - 0.5
            let srcY = (Float(y) + 0.5) * scaleY - 0.5

            let x0 = Int(floor(srcX))
            let y0 = Int(floor(srcY))
            let x1 = min(x0 + 1, image.width - 1)
            let y1 = min(y0 + 1, image.height - 1)

            let fx = srcX - Float(x0)
            let fy = srcY - Float(y0)

            let x0c = max(0, x0)
            let y0c = max(0, y0)

            for c in 0..<3 {
                let p00 = image.pixels[(y0c * image.width + x0c) * 3 + c]
                let p10 = image.pixels[(y0c * image.width + x1) * 3 + c]
                let p01 = image.pixels[(y1 * image.width + x0c) * 3 + c]
                let p11 = image.pixels[(y1 * image.width + x1) * 3 + c]

                let value = (1 - fx) * (1 - fy) * p00 +
                    fx * (1 - fy) * p10 +
                    (1 - fx) * fy * p01 +
                    fx * fy * p11

                resizedPixels[(y * targetWidth + x) * 3 + c] = value
            }
        }
    }

    let scaledFocal = image.focalLengthPx * Float(targetWidth) / Float(image.width)

    return LoadedImage(
        pixels: resizedPixels,
        width: targetWidth,
        height: targetHeight,
        focalLengthPx: scaledFocal
    )
}
