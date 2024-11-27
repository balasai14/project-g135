import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import * as tf from "@tensorflow/tfjs";

const FaceRecognition = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [modelsLoaded, setModelsLoaded] = useState(false);
    const [labelDict, setLabelDict] = useState(null);
    const [recognitionModel, setRecognitionModel] = useState(null);

    useEffect(() => {
        const loadModels = async () => {
            try {
                // Load face-api.js models
                await Promise.all([
                    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
                    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
                    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
                ]);

                // Load the TensorFlow.js model and label dictionary
                const model = await tf.loadLayersModel('/face_recognition_model.h5');
                setRecognitionModel(model);

                const response = await fetch('/label_dict.npy');
                const buffer = await response.arrayBuffer();
                const labels = Object.entries(JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(buffer))));
                setLabelDict(labels);

                setModelsLoaded(true);
                startVideo();
            } catch (error) {
                console.error("Failed to load models:", error);
            }
        };

        loadModels();
    }, []);

    const startVideo = () => {
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then((stream) => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            })
            .catch((error) => console.error("Error accessing webcam:", error));
    };

    const handleVideoPlay = () => {
        const video = videoRef.current;

        setInterval(async () => {
            if (video && modelsLoaded && recognitionModel && labelDict) {
                const detections = await faceapi.detectAllFaces(
                    video,
                    new faceapi.TinyFaceDetectorOptions()
                ).withFaceLandmarks().withFaceDescriptors();

                const canvas = canvasRef.current;
                if (canvas) {
                    const displaySize = { width: video.videoWidth, height: video.videoHeight };
                    faceapi.matchDimensions(canvas, displaySize);

                    const resizedDetections = faceapi.resizeResults(detections, displaySize);
                    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

                    // Recognize faces
                    resizedDetections.forEach(({ descriptor, detection }) => {
                        const embedding = tf.tensor1d(descriptor);
                        const prediction = recognitionModel.predict(embedding.expandDims(0));
                        const predictedLabel = labelDict[prediction.argMax(-1).dataSync()[0]];

                        const { box } = detection;
                        const { x, y, width, height } = box;
                        const ctx = canvas.getContext("2d");
                        ctx.strokeStyle = "blue";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, width, height);

                        ctx.font = "16px Arial";
                        ctx.fillStyle = "blue";
                        ctx.fillText(predictedLabel, x, y - 10);
                    });
                }
            }
        }, 100);
    };

    return (
        <div className="relative">
            <video
                ref={videoRef}
                autoPlay
                muted
                onPlay={handleVideoPlay}
                className="rounded-lg shadow-md w-full"
            ></video>
            <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 rounded-lg"
                style={{ width: "100%", height: "100%" }}
            ></canvas>
            {!modelsLoaded && (
                <p className="text-center text-gray-400 mt-4">Loading models...</p>
            )}
        </div>
    );
};

export default FaceRecognition;
