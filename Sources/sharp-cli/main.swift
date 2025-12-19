import ArgumentParser
import Foundation
import Sharp

@main
struct SharpCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sharp-cli",
        abstract: "SHARP - Monocular 3D Gaussian Prediction (Swift/Metal)",
        version: "0.1.0"
    )

    @Option(name: .shortAndLong, help: "Path to input image or directory")
    var input: String

    @Option(name: .shortAndLong, help: "Path to output directory")
    var output: String = "samples/"

    @Option(name: .long, help: "URL to download CoreML model from (zip file)")
    var modelUrl: String = Sharp.defaultModelURL.absoluteString

    @Option(name: .long, help: "Directory to cache downloaded models")
    var modelCache: String = defaultModelCacheDirectory().path

    @Flag(name: .shortAndLong, help: "Verbose output")
    var verbose: Bool = false

    func run() async throws {
        let totalStart = Date()
        
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: output)
        let modelURL = URL(string: modelUrl)!
        let cacheURL = URL(fileURLWithPath: modelCache)

        var imagePaths: [URL] = []
        if FileManager.default.isDirectory(at: inputURL) {
            imagePaths = try findImages(in: inputURL)
        } else if isSupportedImage(inputURL) {
            imagePaths = [inputURL]
        }

        guard !imagePaths.isEmpty else {
            throw ValidationError("No valid images found in: \(input)")
        }

        print("📷 Found \(imagePaths.count) image(s)")
        print("📁 Output directory: \(outputURL.path)")
        print()

        // Check for cached model
        let sharp: Sharp
        if let cachedModelURL = Sharp.cachedModel(in: cacheURL) {
            print("🔧 Using cached model: \(cachedModelURL.path)")
            let loadStart = Date()
            sharp = try Sharp(modelURL: cachedModelURL)
            let loadElapsed = Date().timeIntervalSince(loadStart)
            print("   Model loaded in \(String(format: "%.2f", loadElapsed))s")
        } else {
            print("🔧 Downloading model from: \(modelURL.absoluteString)")
            print("   Cache directory: \(cacheURL.path)")
            let downloadStart = Date()
            sharp = try await Sharp.download(from: modelURL, to: cacheURL) { progress in
                let percent = Int(progress * 100)
                if percent % 10 == 0 {
                    print("   Downloading: \(percent)%", terminator: "\r")
                    fflush(stdout)
                }
            }
            let downloadElapsed = Date().timeIntervalSince(downloadStart)
            print("   Model downloaded and loaded in \(String(format: "%.2f", downloadElapsed))s")
        }
        print()

        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let options = SharpOptions(verbose: verbose)

        for (idx, imagePath) in imagePaths.enumerated() {
            print("🖼️  [\(idx + 1)/\(imagePaths.count)] Processing: \(imagePath.path)")

            let outputPLY = outputURL.appendingPathComponent(
                imagePath.deletingPathExtension().lastPathComponent + ".ply"
            )

            let imageStart = Date()
            try sharp.convert(
                from: imagePath,
                to: outputPLY,
                options: options
            )
            let imageElapsed = Date().timeIntervalSince(imageStart)

            print("   Output: \(outputPLY.path)")
            print("   Completed in \(String(format: "%.2f", imageElapsed))s")
        }

        let totalElapsed = Date().timeIntervalSince(totalStart)
        print()
        print("✅ Done! Total time: \(String(format: "%.2f", totalElapsed))s")
    }
}

func defaultModelCacheDirectory() -> URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return cacheDir.appendingPathComponent("sharp-cli/models")
}

func findImages(in directory: URL) throws -> [URL] {
    let contents = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: nil
    )
    return contents.filter { isSupportedImage($0) }.sorted { $0.path < $1.path }
}

func isSupportedImage(_ url: URL) -> Bool {
    let ext = url.pathExtension.lowercased()
    return ["jpg", "jpeg", "png", "heic", "heif"].contains(ext)
}

extension FileManager {
    func isDirectory(at url: URL) -> Bool {
        var isDir: ObjCBool = false
        return fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
    }
}
