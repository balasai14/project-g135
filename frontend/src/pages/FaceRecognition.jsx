import { useRef, useEffect, useState } from "react";
import * as faceapi from "face-api.js";

const FaceRecognition = () => {
    const videoRef = useRef(null);
    const [loading, setLoading] = useState(true);
    const [recognizedFace, setRecognizedFace] = useState("No face detected");

    useEffect(() => {
        const loadModels = async () => {
            setLoading(true);
            const CDN_URL = "https://github.com/justadudewhohacks/face-api.js/raw/master/weights";

            // Load models directly from the CDN
            await faceapi.nets.tinyFaceDetector.loadFromUri(CDN_URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(CDN_URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(CDN_URL);
            setLoading(false);
        };

        const startVideo = () => {
            navigator.mediaDevices
                .getUserMedia({ video: true })
                .then((stream) => {
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                    }
                })
                .catch((err) => console.error("Error accessing webcam: ", err));
        };

        loadModels().then(startVideo);

        // Cleanup on component unmount
        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                // eslint-disable-next-line react-hooks/exhaustive-deps
                videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
            }
        };
    }, []);

    const handleVideoPlay = async () => {
        const labeledFaceDescriptors = await loadLabeledImages();
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

        setInterval(async () => {
            if (videoRef.current) {
                const detections = await faceapi
                    .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
                    .withFaceLandmarks()
                    .withFaceDescriptors();

                if (detections.length > 0) {
                    const results = detections.map((d) =>
                        faceMatcher.findBestMatch(d.descriptor)
                    );
                    setRecognizedFace(results[0].toString());
                } else {
                    setRecognizedFace("No face detected");
                }
            }
        }, 1000);
    };

    const loadLabeledImages = () => {
        const labels = ["User"]; // Replace with your label(s)
        return Promise.all(
            labels.map(async (label) => {
                const descriptions = [];
                for (let i = 1; i <= 3; i++) {
                    const img = await faceapi.fetchImage(`/images/${label}/${i}.jpg`);
                    const detections = await faceapi
                        .detectSingleFace(img)
                        .withFaceLandmarks()
                        .withFaceDescriptor();
                    descriptions.push(detections.descriptor);
                }
                return new faceapi.LabeledFaceDescriptors(label, descriptions);
            })
        );
    };

    return (
        <div className="face-recognition">
            {loading ? (
                <p>Loading models...</p>
            ) : (
                <div>
                    <video
                        ref={videoRef}
                        autoPlay
                        muted
                        onPlay={handleVideoPlay}
                        style={{ width: "100%", borderRadius: "10px" }}
                    />
                    <p className="mt-4 text-center text-gray-300">
                        <strong>Recognized Face:</strong> {recognizedFace}
                    </p>
                </div>
            )}
        </div>
    );
};

export default FaceRecognition;
