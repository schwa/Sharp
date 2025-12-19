import Foundation

func sigmoid(_ x: Float) -> Float {
    1.0 / (1.0 + exp(-x))
}

func inverseSigmoid(_ x: Float) -> Float {
    log(x / (1.0 - x))
}

func clamp<T: Comparable>(_ value: T, min minValue: T, max maxValue: T) -> T {
    min(max(value, minValue), maxValue)
}
