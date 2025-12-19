import ArgumentParser
import Foundation
import Sharp

@main
struct SharpCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sharp-cli",
        abstract: "SHARP - Monocular 3D Gaussian Prediction (Swift/Metal)",
        version: "0.1.0"
    )

    @Option(name: .shortAndLong, help: "Path to input image or directory")
    var input: String

    @Option(name: .shortAndLong, help: "Path to output directory")
    var output: String = "samples/"

    @Option(name: .shortAndLong, help: "Path to CoreML model (.mlpackage or .mlmodelc)")
    var model: String

    @Flag(name: .shortAndLong, help: "Verbose output")
    var verbose: Bool = false

    func run() throws {
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: output)
        let modelURL = URL(fileURLWithPath: model)

        var imagePaths: [URL] = []
        if FileManager.default.isDirectory(at: inputURL) {
            imagePaths = try findImages(in: inputURL)
        } else if isSupportedImage(inputURL) {
            imagePaths = [inputURL]
        }

        guard !imagePaths.isEmpty else {
            throw ValidationError("No valid images found in: \(input)")
        }

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ValidationError("Model not found: \(model)")
        }

        print("📷 Found \(imagePaths.count) image(s)")
        print("📁 Output: \(output)")
        print()

        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let options = SharpOptions(verbose: verbose)

        for (idx, imagePath) in imagePaths.enumerated() {
            print("🖼️  [\(idx + 1)/\(imagePaths.count)] \(imagePath.lastPathComponent)")

            let outputPLY = outputURL.appendingPathComponent(
                imagePath.deletingPathExtension().lastPathComponent + ".ply"
            )

            let startTime = Date()
            try generateGaussianSplat(
                from: imagePath,
                model: modelURL,
                options: options,
                to: outputPLY
            )
            let elapsed = Date().timeIntervalSince(startTime)

            print("   Done in \(String(format: "%.2f", elapsed))s")
        }

        print()
        print("✅ Done!")
    }
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
