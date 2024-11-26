import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";

const FaceRecognition = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [modelsLoaded, setModelsLoaded] = useState(false);

    useEffect(() => {
        const loadModels = async () => {
            try {
                // Use the correct path for model files in the public folder
                await Promise.all([
                    faceapi.nets.tinyFaceDetector.loadFromUri('/models/tiny_face_detector_model-weights_manifest.json'),
                    faceapi.nets.faceLandmark68Net.loadFromUri('/models/face_landmark_68_model-weights_manifest.json'),
                    faceapi.nets.faceRecognitionNet.loadFromUri('/models/face_recognition_model-weights_manifest.json'),
                ]);
                setModelsLoaded(true);
                startVideo();
            } catch (error) {
                console.error("Failed to load face recognition models:", error);
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
            if (video && modelsLoaded) {
                const detections = await faceapi.detectAllFaces(
                    video,
                    new faceapi.TinyFaceDetectorOptions()
                );

                const canvas = canvasRef.current;
                if (canvas) {
                    const displaySize = { width: video.videoWidth, height: video.videoHeight };
                    faceapi.matchDimensions(canvas, displaySize);

                    const resizedDetections = faceapi.resizeResults(detections, displaySize);
                    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
                    faceapi.draw.drawDetections(canvas, resizedDetections);
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
