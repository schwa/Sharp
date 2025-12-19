#if !arch(x86_64)
import CoreML
import Foundation
import Metal
import simd

class MetalProcessor {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let extractAndUnprojectPipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDevice
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MetalError.noCommandQueue
        }
        self.commandQueue = queue

        let library: MTLLibrary
        do {
            guard let shaderURL = Bundle.module.url(forResource: "Shaders", withExtension: "metal"),
                  let shaderSource = try? String(contentsOf: shaderURL) else {
                throw MetalError.shaderCompilationFailed("Could not load Shaders.metal")
            }
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            throw MetalError.shaderCompilationFailed(error.localizedDescription)
        }

        guard let function = library.makeFunction(name: "extractAndUnproject") else {
            throw MetalError.functionNotFound("extractAndUnproject")
        }

        self.extractAndUnprojectPipeline = try device.makeComputePipelineState(function: function)
    }

    func extractGaussians(
        meanArray: MLMultiArray,
        scaleArray: MLMultiArray,
        quatArray: MLMultiArray,
        colorArray: MLMultiArray,
        opacityArray: MLMultiArray
    ) throws -> Gaussians3D {
        let count = meanArray.shape[1].intValue

        let meanStride = UInt32(meanArray.strides[1].intValue)
        let scaleStride = UInt32(scaleArray.strides[1].intValue)
        let quatStride = UInt32(quatArray.strides[1].intValue)
        let colorStride = UInt32(colorArray.strides[1].intValue)
        let opacityStride = UInt32(opacityArray.strides[1].intValue)

        let meanBuffer = device.makeBuffer(
            bytes: meanArray.dataPointer,
            length: meanArray.strides[0].intValue * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        let scaleBuffer = device.makeBuffer(
            bytes: scaleArray.dataPointer,
            length: scaleArray.strides[0].intValue * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        let quatBuffer = device.makeBuffer(
            bytes: quatArray.dataPointer,
            length: quatArray.strides[0].intValue * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        let colorBuffer = device.makeBuffer(
            bytes: colorArray.dataPointer,
            length: colorArray.strides[0].intValue * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        let opacityBuffer = device.makeBuffer(
            bytes: opacityArray.dataPointer,
            length: opacityArray.strides[0].intValue * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        let outMeansBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.size, options: .storageModeShared)!
        let outScalesBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.size, options: .storageModeShared)!
        let outQuatsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.size, options: .storageModeShared)!
        let outColorsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.size, options: .storageModeShared)!
        let outOpacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }

        encoder.setComputePipelineState(extractAndUnprojectPipeline)

        encoder.setBuffer(meanBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(quatBuffer, offset: 0, index: 2)
        encoder.setBuffer(colorBuffer, offset: 0, index: 3)
        encoder.setBuffer(opacityBuffer, offset: 0, index: 4)
        encoder.setBuffer(outMeansBuffer, offset: 0, index: 5)
        encoder.setBuffer(outScalesBuffer, offset: 0, index: 6)
        encoder.setBuffer(outQuatsBuffer, offset: 0, index: 7)
        encoder.setBuffer(outColorsBuffer, offset: 0, index: 8)
        encoder.setBuffer(outOpacitiesBuffer, offset: 0, index: 9)

        var meanStrideVar = meanStride
        var scaleStrideVar = scaleStride
        var quatStrideVar = quatStride
        var colorStrideVar = colorStride
        var opacityStrideVar = opacityStride

        encoder.setBytes(&meanStrideVar, length: MemoryLayout<UInt32>.size, index: 10)
        encoder.setBytes(&scaleStrideVar, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&quatStrideVar, length: MemoryLayout<UInt32>.size, index: 12)
        encoder.setBytes(&colorStrideVar, length: MemoryLayout<UInt32>.size, index: 13)
        encoder.setBytes(&opacityStrideVar, length: MemoryLayout<UInt32>.size, index: 14)

        let threadGroupSize = min(extractAndUnprojectPipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let meansPtr = outMeansBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        let scalesPtr = outScalesBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        let quatsPtr = outQuatsBuffer.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let colorsPtr = outColorsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        let opacitiesPtr = outOpacitiesBuffer.contents().assumingMemoryBound(to: Float.self)

        let means = Array(UnsafeBufferPointer(start: meansPtr, count: count))
        let scales = Array(UnsafeBufferPointer(start: scalesPtr, count: count))
        let quats = Array(UnsafeBufferPointer(start: quatsPtr, count: count))
        let colors = Array(UnsafeBufferPointer(start: colorsPtr, count: count))
        let opacities = Array(UnsafeBufferPointer(start: opacitiesPtr, count: count))

        return Gaussians3D(
            meanVectors: means,
            singularValues: scales,
            quaternions: quats,
            colors: colors,
            opacities: opacities
        )
    }
}

enum MetalError: Error, LocalizedError {
    case noDevice
    case noCommandQueue
    case shaderCompilationFailed(String)
    case functionNotFound(String)
    case encoderCreationFailed

    var errorDescription: String? {
        switch self {
        case .noDevice: return "No Metal device available"
        case .noCommandQueue: return "Failed to create command queue"
        case .shaderCompilationFailed(let msg): return "Shader compilation failed: \(msg)"
        case .functionNotFound(let name): return "Metal function not found: \(name)"
        case .encoderCreationFailed: return "Failed to create compute encoder"
        }
    }
}
#endif
