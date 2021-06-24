import Foundation

struct InferenceResult {
    let score: Float
    let label: String
}

class Predictor {

    private var isRunning: Bool = false

    private lazy var module: VisionTorchModule = {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "pt"),
            let module = VisionTorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Failed to load model!")
        }
    }()

    private var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Label file was not found.")
        }
    }()

    func predict(_ buffer: [Float32], resultCount: Int) -> [InferenceResult]? {
        if isRunning {
            return nil
        }
        isRunning = true
        var tensorBuffer = buffer

        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&tensorBuffer)) else {
            return nil
        }

        isRunning = false
        return topK(scores: outputs, labels: labels, count: resultCount)
    }

    func topK(scores: [NSNumber], labels: [String], count: Int) -> [InferenceResult] {
        let zippedResults = zip(labels.indices, scores)
        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(count)
        return sortedResults.map { InferenceResult(score: $0.1.floatValue, label: labels[$0.0]) }
    }



}
