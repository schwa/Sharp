import Foundation
import ZIPFoundation

// MARK: - Options

public struct SharpOptions: Sendable {
    public var verbose: Bool
    public var colorSpace: ColorSpace

    public init(
        verbose: Bool = false,
        colorSpace: ColorSpace = .linearRGB
    ) {
        self.verbose = verbose
        self.colorSpace = colorSpace
    }
}

// MARK: - Errors

public enum SharpError: Error, LocalizedError {
    case downloadFailed(String)
    case unzipFailed(String)
    case modelNotFound(String)
    case invalidArchive(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        case .unzipFailed(let msg): return "Unzip failed: \(msg)"
        case .modelNotFound(let msg): return "Model not found: \(msg)"
        case .invalidArchive(let msg): return "Invalid archive: \(msg)"
        }
    }
}

// MARK: - Sharp

public struct Sharp: Sendable {
    private let predictor: SharpPredictor

    /// Default model URL (hosted on Hugging Face)
    public static let defaultModelURL = URL(string: "https://huggingface.co/jwight/spark/resolve/main/SharpPredictor.mlmodelc.zip")!

    /// Initialize with a local CoreML model URL (.mlpackage or .mlmodelc)
    public init(modelURL: URL) throws {
        self.predictor = try SharpPredictor(modelURL: modelURL)
    }

    /// Check if a model is already cached in the given directory
    /// - Parameter directory: Directory to check for cached model
    /// - Returns: URL to the cached model if found, nil otherwise
    public static func cachedModel(in directory: URL) -> URL? {
        try? findModel(in: directory)
    }

    /// Download a model from a URL, unzip it, and initialize
    /// - Parameters:
    ///   - url: Remote URL to download the model from (expects a .zip file). Defaults to `defaultModelURL`.
    ///   - destinationDirectory: Local directory to save the unzipped model
    ///   - progress: Optional progress callback (0.0 to 1.0)
    /// - Returns: Initialized Sharp instance
    public static func download(
        from url: URL = defaultModelURL,
        to destinationDirectory: URL,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Sharp {
        let modelURL = try await downloadAndUnzip(
            from: url,
            to: destinationDirectory,
            progress: progress
        )
        return try Sharp(modelURL: modelURL)
    }

    /// Convert an image to 3D Gaussian splats and save as PLY
    /// - Parameters:
    ///   - imageURL: URL of the input image
    ///   - outputURL: URL to save the output PLY file
    ///   - options: Conversion options
    public func convert(
        from imageURL: URL,
        to outputURL: URL,
        options: SharpOptions = SharpOptions()
    ) throws {
        let image = try loadImage(from: imageURL)
        let gaussians = try predictor.predict(image: image, verbose: options.verbose)

        try savePLY(
            gaussians,
            focalLengthPx: image.focalLengthPx,
            imageWidth: image.width,
            imageHeight: image.height,
            to: outputURL,
            colorSpace: options.colorSpace
        )
    }

    /// Convert an image to 3D Gaussian splats
    /// - Parameters:
    ///   - imageURL: URL of the input image
    ///   - options: Conversion options
    /// - Returns: The generated Gaussians and image metadata
    public func convert(
        from imageURL: URL,
        options: SharpOptions = SharpOptions()
    ) throws -> (gaussians: Gaussians3D, image: LoadedImage) {
        let image = try loadImage(from: imageURL)
        let gaussians = try predictor.predict(image: image, verbose: options.verbose)
        return (gaussians, image)
    }
}

// MARK: - Download & Unzip

extension Sharp {
    private static func downloadAndUnzip(
        from url: URL,
        to destinationDirectory: URL,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> URL {
        // Create destination directory
        try FileManager.default.createDirectory(
            at: destinationDirectory,
            withIntermediateDirectories: true
        )

        // Check if model already exists (cached)
        if let existingModel = try? findModel(in: destinationDirectory) {
            progress?(1.0)
            return existingModel
        }

        // Download the file
        let tempZipURL = destinationDirectory.appendingPathComponent("model_download.zip")

        defer {
            try? FileManager.default.removeItem(at: tempZipURL)
        }

        try await downloadFile(from: url, to: tempZipURL, progress: progress)

        // Unzip
        let unzippedURL = try unzip(fileAt: tempZipURL, to: destinationDirectory)

        // Find the model in the unzipped contents
        let modelURL = try findModel(in: unzippedURL)

        return modelURL
    }

    private static func downloadFile(
        from url: URL,
        to destination: URL,
        progress: (@Sendable (Double) -> Void)?
    ) async throws {
        let delegate = DownloadProgressDelegate(progress: progress)
        let (localURL, response) = try await URLSession.shared.download(for: URLRequest(url: url), delegate: delegate)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw SharpError.downloadFailed("HTTP status code: \(statusCode)")
        }

        // Move downloaded file to destination
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: localURL, to: destination)
    }

    private static func unzip(fileAt zipURL: URL, to destinationDirectory: URL) throws -> URL {
        do {
            try FileManager.default.unzipItem(at: zipURL, to: destinationDirectory)
        } catch {
            throw SharpError.unzipFailed(error.localizedDescription)
        }
        return destinationDirectory
    }

    private static func findModel(in directory: URL) throws -> URL {
        // Look for .mlmodelc first (compiled, faster to load)
        if let modelURL = try findFile(in: directory, withExtension: "mlmodelc") {
            return modelURL
        }

        // Then look for .mlpackage
        if let modelURL = try findFile(in: directory, withExtension: "mlpackage") {
            return modelURL
        }

        throw SharpError.modelNotFound(
            "No .mlmodelc or .mlpackage found in \(directory.path)"
        )
    }

    private static func findFile(in directory: URL, withExtension ext: String) throws -> URL? {
        let fileManager = FileManager.default

        // Check direct children first
        let contents = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey]
        )

        for item in contents {
            if item.pathExtension == ext {
                return item
            }
        }

        // Check one level deeper (common for zip archives)
        for item in contents {
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: item.path, isDirectory: &isDirectory),
               isDirectory.boolValue {
                let subContents = try fileManager.contentsOfDirectory(
                    at: item,
                    includingPropertiesForKeys: nil
                )
                for subItem in subContents {
                    if subItem.pathExtension == ext {
                        return subItem
                    }
                }
            }
        }

        return nil
    }
}

// MARK: - Download Progress Delegate

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate {
    private let progress: (@Sendable (Double) -> Void)?

    init(progress: (@Sendable (Double) -> Void)?) {
        self.progress = progress
        super.init()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Required but handled by async API
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        progress?(fraction)
    }
}

// MARK: - Legacy API (backwards compatibility)

/// Generate 3D Gaussian splats from an image
/// - Note: Consider using `Sharp` struct directly for better performance when processing multiple images
@available(*, deprecated, message: "Use Sharp struct instead")
public func generateGaussianSplat(
    from imageURL: URL,
    model modelURL: URL,
    options: SharpOptions = SharpOptions(),
    to outputURL: URL
) throws {
    let sharp = try Sharp(modelURL: modelURL)
    try sharp.convert(from: imageURL, to: outputURL, options: options)
}
