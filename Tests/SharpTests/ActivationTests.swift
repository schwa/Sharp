@testable import Sharp
import Testing

@Suite("Activation Function Tests")
struct ActivationTests {
    @Test("Sigmoid range")
    func sigmoidRange() {
        #expect(sigmoid(-100) >= 0)
        #expect(sigmoid(-100) < 0.01)
        #expect(sigmoid(100) <= 1)
        #expect(sigmoid(100) > 0.99)
        #expect(abs(sigmoid(0) - 0.5) < 1e-6)

        #expect(sigmoid(-10) > 0)
        #expect(sigmoid(10) < 1)
    }

    @Test("Sigmoid inverse roundtrip")
    func sigmoidInverseRoundtrip() {
        let testValues: [Float] = [0.1, 0.3, 0.5, 0.7, 0.9]

        for x in testValues {
            let logit = inverseSigmoid(x)
            let roundtrip = sigmoid(logit)
            #expect(abs(roundtrip - x) < 1e-5, "Sigmoid roundtrip failed for \(x)")
        }
    }

    @Test("Clamp functions")
    func clampFunctions() {
        #expect(clamp(5, min: 0, max: 10) == 5)
        #expect(clamp(-5, min: 0, max: 10) == 0)
        #expect(clamp(15, min: 0, max: 10) == 10)
    }
}
