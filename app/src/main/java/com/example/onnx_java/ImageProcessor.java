package com.example.onnx_java;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import android.graphics.Bitmap;

public class ImageProcessor {

    public static float[][][] preprocessImage(Bitmap bitmap, int inputWidth, int inputHeight) {
        // Convert Bitmap to Mat
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);

        // Convert BGR to RGB
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);

        // Resize the image
        Imgproc.resize(img, img, new Size(inputWidth, inputHeight));

        // Normalize and convert to float array
        float[][][] imageData = new float[3][inputHeight][inputWidth];
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                double[] pixel = img.get(y, x);
                // Normalize and assign to the correct channel
                imageData[0][y][x] = (float) (pixel[0] / 255.0); // Red channel
                imageData[1][y][x] = (float) (pixel[1] / 255.0); // Green channel
                imageData[2][y][x] = (float) (pixel[2] / 255.0); // Blue channel
            }
        }

        return imageData;
    }
}

