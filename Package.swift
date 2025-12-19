// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Sharp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "Sharp",
            targets: ["Sharp"]
        ),
        .executable(
            name: "sharp-cli",
            targets: ["sharp-cli"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0")
    ],
    targets: [
        .target(
            name: "Sharp",
            dependencies: [],
            path: "Sources/Sharp",
            resources: [
                .process("Shaders.metal")
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK")
            ]
        ),
        .executableTarget(
            name: "sharp-cli",
            dependencies: [
                "Sharp",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "Sources/sharp-cli"
        ),
        .testTarget(
            name: "SharpTests",
            dependencies: ["Sharp"],
            path: "Tests/SharpTests"
        )
    ]
)
