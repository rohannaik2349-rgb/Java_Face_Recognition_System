package rohan.in;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import org.opencv.videoio.VideoCapture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    private static final int SIZE =620;
    private static final double THRESHOLD = 0.363;

    static void main() {
        OpenCV.loadLocally();

        Path modelDir = Path.of("models");
        Path yunetPath = modelDir.resolve("face_detection_yunet_2023mar.onnx");
        Path sfacePath = modelDir.resolve("face_recognition_sface_2021dec_int8.onnx");

        System.out.println("YuNet path: "+yunetPath);
        System.out.println("Yunet exists : "+Files.exists(yunetPath));

        System.out.println("sface path: "+sfacePath);
        System.out.println("sFace exists : "+Files.exists(sfacePath));

        if(!Files.exists(yunetPath) || !Files.exists(sfacePath)){
            System.out.println("Check the missing model file");
            return;
        }

        FaceDetectorYN detector = FaceDetectorYN.create(
                yunetPath.toString(),
                "",
                new Size(620, 620),
                0.6f,
                0.3f,
                5000
        );
        FaceRecognizerSF recognizer = FaceRecognizerSF.create(sfacePath.toString(), "");

        List<Mat> knownFeatures = new ArrayList<>();
        List<String> knownName = new ArrayList<>();

        String[] photos = {
                "Known_faces/Rohan/Rohan.jpeg",
                "Known_faces/Rohan/Rohan2.jpeg",
                "Known_faces/Rohan/rohan3.jpeg",
                "Known_faces/Rohan/rohan4.jpeg"
        };

        for (String photoPath : photos) {
            Mat image = Imgcodecs.imread(photoPath);

            if (image.empty()) {
                System.out.println("Cannot read photo: " + photoPath);
                continue;
            }

            Mat resized = new Mat();
            Imgproc.resize(image, resized, new Size(SIZE, SIZE));
            detector.setInputSize(new Size(SIZE, SIZE));

            Mat faces = new Mat();
            detector.detect(resized, faces);

            if (faces.rows() == 0) {
                System.out.println("No face found in: " + photoPath);
                continue;
            }

            Mat aligned = new Mat();
            Mat feature = new Mat();

            recognizer.alignCrop(resized, faces.row(0), aligned);
            recognizer.feature(aligned, feature);

            knownFeatures.add(feature);
            knownName.add("Rohan");

            System.out.println("Loaded reference: " + photoPath);
        }

            if (knownFeatures.isEmpty()) {
                System.out.println("No reference faces loaded Check photos.");
            }
            VideoCapture camera = new VideoCapture(0);
            if (!camera.isOpened()) {
                System.out.println("Cannot open webcam");
                return;
            }


            Mat frame = new Mat();
            Mat faces = new Mat();
            Mat resizFrame = new Mat();

            System.out.println("Recognition started.Press ESC to exit");

            while (camera.read(frame)) {
                if (frame.empty()) {
                    continue;
                }
                Imgproc.resize(frame, resizFrame, new Size(SIZE, SIZE));
                detector.setInputSize(new Size(SIZE, SIZE));
                detector.detect(resizFrame, faces);
                for (int i = 0; i < faces.rows(); i++) {
                    Mat faceRow = faces.row(i);

                    int x = (int) faces.get(i, 0)[0];
                    int y = (int) faces.get(i, 1)[0];
                    int w = (int) faces.get(i, 2)[0];
                    int h = (int) faces.get(i, 3)[0];

                    Mat aligned = new Mat();
                    Mat feature = new Mat();

                    recognizer.alignCrop(resizFrame, faceRow, aligned);
                    recognizer.feature(aligned, feature);

                    String bestName = "Unknown";
                    String name = "UnKnown";
                    double bestScore = -1.0;


                    for (int j = 0; j < knownFeatures.size(); j++) {
                        double score = recognizer.match(
                                feature,
                                knownFeatures.get(j),
                                FaceRecognizerSF.FR_COSINE
                        );
                        if (Double.isFinite(score) && score > bestScore) {
                            bestScore = score;
                            bestName = knownName.get(j);
                        }

                        if (score > bestScore) {
                            bestScore = score;
                            bestName = knownName.get(j);
                        }

                    }

                        if (bestScore >= THRESHOLD) {
                            name = bestName;
                        }



                        Imgproc.rectangle(
                                resizFrame,
                                new Point(x, y),
                                new Point(x + w, y + h),
                                new Scalar(0, 255, 0),
                                3
                        );

                        Imgproc.putText(
                                resizFrame,
                                name + "(" + String.format("%.2f", bestScore) + ")",
                                new Point(x, Math.max(y - 8, 20)),
                                Imgproc.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                new Scalar(0, 255, 0),
                                2
                        );
                        aligned.release();
                        feature.release();
                    }

                    HighGui.namedWindow("Java Face Detection",HighGui.WINDOW_NORMAL);
                    HighGui.resizeWindow("Java Face Detection",1000,700);
                    HighGui.imshow("Java Face Detection", resizFrame);

                    if (HighGui.waitKey(1) == 27) {
                        break;
                    }
                }

                camera.release();

                HighGui.destroyAllWindows();

                for (Mat feature : knownFeatures) {
                    feature.release();
                }
            }
        }
