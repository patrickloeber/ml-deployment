import UIKit
import AVFoundation

class ViewController: UIViewController {

    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var preview: UIView!

    private var captureSession: AVCaptureSession!
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    private var videoOutput = AVCaptureVideoDataOutput()
    private var sessionQueue = DispatchQueue(label: "session")
    private var bufferQueue = DispatchQueue(label: "buffer")
    private var predictor = Predictor()

    override func viewDidLoad() {
        super.viewDidLoad()

        startSession()
    }

    private func startSession() {
        captureSession = AVCaptureSession()
        sessionQueue.async {
            self.captureSession.sessionPreset = .high
            self.captureSession.beginConfiguration()
            self.configureCamera()
            self.configureOutput()
            self.captureSession.commitConfiguration()

            self.prepare {
                if $0, !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                }
            }
        }
    }

    private func setupLivePreview() {
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        videoPreviewLayer.videoGravity = .resizeAspect
        videoPreviewLayer.connection?.videoOrientation = .portrait
        preview.layer.addSublayer(videoPreviewLayer)

        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
            DispatchQueue.main.async {
                self.videoPreviewLayer.frame = self.preview.bounds
            }
        }
    }

    func configureCamera() {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }

        let input = try! AVCaptureDeviceInput(device: camera)
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
    }

    func configureOutput() {
        videoOutput.setSampleBufferDelegate(self, queue: bufferQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [String(kCVPixelBufferPixelFormatTypeKey): kCMPixelFormat_32BGRA]
        guard captureSession.canAddOutput(videoOutput) else {
            return
        }
        captureSession.addOutput(videoOutput)
        DispatchQueue.main.async {
            self.setupLivePreview()
        }
    }

    private func prepare(_ completionHandler: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .notDetermined {
            AVCaptureDevice.requestAccess(for: .video) {
                completionHandler($0)
            }
            return
        }
        completionHandler(status == .authorized)
    }

    func process(buffer: [Float]?) {
        guard let pixelBuffer = buffer else { return }

        let result = predictor.predict(pixelBuffer, resultCount: 2)

        DispatchQueue.main.async {
            guard let isEmpty = result?.first?.label.isEmpty, !isEmpty else { return }
            self.resultLabel.text = result?.first?.label
        }
    }

}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        connection.videoOrientation = .portrait
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        guard let normalizedBuffer = pixelBuffer.normalized(224, 224) else {
            return
        }

        process(buffer: normalizedBuffer)
    }
}
