import Foundation

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

public func generateGaussianSplat(
    from imageURL: URL,
    model modelURL: URL,
    options: SharpOptions = SharpOptions(),
    to outputURL: URL
) throws {
    let image = try loadImage(from: imageURL)
    let predictor = try SharpPredictor(modelURL: modelURL)
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
