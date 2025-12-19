import CoreGraphics
import CoreML
import Foundation
import simd

enum SharpPredictorError: Error, LocalizedError {
    case modelLoadFailed(String)
    case predictionFailed(String)
    case invalidInput(String)
    case outputProcessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .predictionFailed(let msg): return "Prediction failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .outputProcessingFailed(let msg): return "Output processing failed: \(msg)"
        }
    }
}

/// SHARP predictor using CoreML
class SharpPredictor {
    private let model: MLModel
    private let metalProcessor: MetalProcessor?

    /// Internal resolution expected by the model
    static let internalResolution = 1_536

    /// Initialize with a CoreML model URL (.mlpackage or .mlmodelc)
    init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        do {
            let compiledURL: URL
            if modelURL.pathExtension == "mlmodelc" {
                compiledURL = modelURL
            } else {
                compiledURL = try MLModel.compileModel(at: modelURL)
            }

            self.model = try MLModel(contentsOf: compiledURL, configuration: config)
            self.metalProcessor = try? MetalProcessor()
        } catch {
            throw SharpPredictorError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Predict 3D Gaussians from an image
    func predict(image: LoadedImage, verbose _: Bool = false) throws -> Gaussians3D {
        let resized = resizeImage(image, to: Self.internalResolution, targetHeight: Self.internalResolution)
        let flippedV = flipImageVertically(resized)
        let flipped = flipImageHorizontally(flippedV)

        let imageArray = try createImageMultiArray(from: flipped)

        let disparityFactor = image.focalLengthPx / Float(image.width)
        let disparityArray = try MLMultiArray(shape: [1], dataType: .float32)
        disparityArray[0] = NSNumber(value: disparityFactor)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: imageArray),
            "disparity_factor": MLFeatureValue(multiArray: disparityArray)
        ])

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: input)
        } catch {
            throw SharpPredictorError.predictionFailed(error.localizedDescription)
        }

        guard let meanArray = output.featureValue(for: "mean_vectors")?.multiArrayValue,
              let scaleArray = output.featureValue(for: "singular_values")?.multiArrayValue,
              let quatArray = output.featureValue(for: "quaternions")?.multiArrayValue,
              let colorArray = output.featureValue(for: "colors")?.multiArrayValue,
              let opacityArray = output.featureValue(for: "opacities")?.multiArrayValue else {
            throw SharpPredictorError.outputProcessingFailed("Missing output tensors")
        }

        let scaleX = Float(Self.internalResolution) / Float(image.width)
        let scaleY = Float(Self.internalResolution) / Float(image.height)
        let intrinsics = CameraIntrinsics(
            fx: image.focalLengthPx * scaleX,
            fy: image.focalLengthPx * scaleY,
            cx: Float(Self.internalResolution) / 2.0,
            cy: Float(Self.internalResolution) / 2.0
        )

        let gaussiansNDC: Gaussians3D
        if let metal = metalProcessor {
            gaussiansNDC = try metal.extractGaussians(
                meanArray: meanArray,
                scaleArray: scaleArray,
                quatArray: quatArray,
                colorArray: colorArray,
                opacityArray: opacityArray
            )
        } else {
            gaussiansNDC = try extractGaussiansCPU(
                meanArray: meanArray,
                scaleArray: scaleArray,
                quatArray: quatArray,
                colorArray: colorArray,
                opacityArray: opacityArray
            )
        }

        return unprojectGaussians(
            gaussiansNDC,
            intrinsics: intrinsics,
            imageWidth: Self.internalResolution,
            imageHeight: Self.internalResolution
        )
    }

    private func createImageMultiArray(from image: LoadedImage) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: image.height), NSNumber(value: image.width)],
            dataType: .float32
        )

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let hw = image.height * image.width

        for y in 0..<image.height {
            for x in 0..<image.width {
                let srcIdx = (y * image.width + x) * 3
                let dstIdx = y * image.width + x
                ptr[dstIdx] = image.pixels[srcIdx]
                ptr[hw + dstIdx] = image.pixels[srcIdx + 1]
                ptr[2 * hw + dstIdx] = image.pixels[srcIdx + 2]
            }
        }

        return array
    }

    private func extractGaussiansCPU(
        meanArray: MLMultiArray,
        scaleArray: MLMultiArray,
        quatArray: MLMultiArray,
        colorArray: MLMultiArray,
        opacityArray: MLMultiArray
    ) throws -> Gaussians3D {
        let count = meanArray.shape[1].intValue

        let meanStride = meanArray.strides[1].intValue
        let scaleStride = scaleArray.strides[1].intValue
        let quatStride = quatArray.strides[1].intValue
        let colorStride = colorArray.strides[1].intValue
        let opacityStride = opacityArray.strides[1].intValue

        let meanPtr = meanArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let scalePtr = scaleArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let quatPtr = quatArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let colorPtr = colorArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let opacityPtr = opacityArray.dataPointer.assumingMemoryBound(to: Float16.self)

        var means = [SIMD3<Float>](repeating: .zero, count: count)
        var scales = [SIMD3<Float>](repeating: .zero, count: count)
        var quats = [SIMD4<Float>](repeating: .zero, count: count)
        var colors = [SIMD3<Float>](repeating: .zero, count: count)
        var opacities = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let mb = i * meanStride
            means[i] = SIMD3(Float(meanPtr[mb]), Float(meanPtr[mb + 1]), Float(meanPtr[mb + 2]))

            let sb = i * scaleStride
            scales[i] = SIMD3(Float(scalePtr[sb]), Float(scalePtr[sb + 1]), Float(scalePtr[sb + 2]))

            let qb = i * quatStride
            quats[i] = SIMD4(Float(quatPtr[qb]), Float(quatPtr[qb + 1]), Float(quatPtr[qb + 2]), Float(quatPtr[qb + 3]))

            let cb = i * colorStride
            colors[i] = SIMD3(Float(colorPtr[cb]), Float(colorPtr[cb + 1]), Float(colorPtr[cb + 2]))

            opacities[i] = Float(opacityPtr[i * opacityStride])
        }

        return Gaussians3D(
            meanVectors: means,
            singularValues: scales,
            quaternions: quats,
            colors: colors,
            opacities: opacities
        )
    }
}
