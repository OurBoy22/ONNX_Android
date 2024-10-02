package com.example.onnx_java;

import static com.example.onnx_java.PostProcessor.processOutput;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import ai.onnxruntime.*;
import ai.onnxruntime.providers.NNAPIFlags;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

public class YOLOv8ONNX {
    private static OrtEnvironment env;
    private OrtSession session;
    static private int inputWidth = 640;  // Set your input width
    static private int inputHeight = 640; // Set your input height
    static OrtSession.SessionOptions sessionOptions;

    static ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 3 * inputWidth * inputHeight);

    FloatBuffer imageDataBuffer;

    OnnxTensor inputTensor;

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
        env = OrtEnvironment.getEnvironment();
        sessionOptions = new OrtSession.SessionOptions();
    }


    public YOLOv8ONNX(String modelPath, Context context) throws OrtException, IOException {
        byte[] modelData = loadModelFromAssets(context, modelPath);  // Load the ONNX model file



        // Enable NNAPI as the execution provider


//        sessionOptions.addNnapi();
//            sessionOptions.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED));
         try {
            EnumSet<NNAPIFlags> nnapiFlags = EnumSet.noneOf(NNAPIFlags.class);
//            nnapiFlags.add(NNAPIFlags.CPU_DISABLED); // Optional flag to disable CPU
//            sessionOptions.addNnapi(nnapiFlags); // Add NNAPI provider with flags
//            sessionOptions.addNnapi();
        } catch (Exception e) {
            System.err.println("Failed to add NNAPI execution provider: " + e.getMessage());
            // Handle exceptions or fall back to default provider
        }
        session = env.createSession(modelData, sessionOptions);
//        System.out.println("Execution providers: " + session.getExecutionProviderNames());
    }
    // Method to load the ONNX model file from the assets folder
    byte[] loadModelFromAssets(Context context, String fileName) throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream inputStream = assetManager.open(fileName);

        // Read the file into a byte array
        int availableBytes = inputStream.available();
        byte[] buffer = new byte[availableBytes];
        inputStream.read(buffer);
        inputStream.close();

        return buffer;
    }

    // Method to load the image from the assets folder
    public Bitmap loadImageFromAssets(Context context, String fileName) throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream inputStream = assetManager.open(fileName);
        return BitmapFactory.decodeStream(inputStream);
    }

    // Method to preprocess the image

    public Bitmap convertBufferToBitmap(FloatBuffer floatBuffer, int width, int height) {
        // Create a new Mat object
        Mat imgMat = new Mat(height, width, CvType.CV_32FC3);

        // Copy the FloatBuffer data to the Mat object
        float[] imgData = new float[3 * width * height];
        floatBuffer.get(imgData);
        imgMat.put(0, 0, imgData);

        // Convert the Mat object to a Bitmap
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imgMat, bitmap);

        return bitmap;
    }

    public Bitmap preprocessImageBitmap(Bitmap bitmap){
        // Convert Bitmap to Mat (OpenCV object)

        // start timer

        Mat imgMat = new Mat();
        Utils.bitmapToMat(bitmap, imgMat);

        // Convert BGR to RGB
        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_BGR2RGB);

        // Resize image
        Imgproc.resize(imgMat, imgMat, new Size(inputWidth, inputHeight));

        // Normalize image data to [0, 1] range
        imgMat.convertTo(imgMat, CvType.CV_32FC3, 1.0 / 255.0);

        // Convert the Mat object to a Bitmap
        Mat imgMat8Bit = new Mat();
        imgMat.convertTo(imgMat8Bit, CvType.CV_8UC3, 255.0);  // Multiply by 255 to restore range [0, 255]

        Bitmap bitmap_out = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888);


        // Use the correctly initialized bitmap_out, not the original bitmap
        Utils.matToBitmap(imgMat8Bit, bitmap_out);



        return bitmap_out; // Return the processed bitmap
    }
    public FloatBuffer preprocessImage(Bitmap bitmap) {
        // Convert Bitmap to Mat (OpenCV object)
        Mat imgMat = new Mat();
        Utils.bitmapToMat(bitmap, imgMat);

        // Convert BGR to RGB
        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_BGR2RGB);

        // Resize image
        Imgproc.resize(imgMat, imgMat, new Size(inputWidth, inputHeight));

        // Normalize image data to [0, 1] range
        imgMat.convertTo(imgMat, CvType.CV_32FC3, 1.0 / 255.0);

        // Prepare a FloatBuffer to hold the image data in CHW format
        int channels = 3; // RGB channels
        // Create a reusable buffer for the tensor
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 3 * inputWidth * inputHeight);
        byteBuffer.order(ByteOrder.nativeOrder());
        FloatBuffer imageDataBuffer = byteBuffer.asFloatBuffer();
//        FloatBuffer imageDataBuffer = FloatBuffer.allocate(channels * inputHeight * inputWidth);

        // Retrieve image data in one go

        float[] imgData = new float[channels * inputHeight * inputWidth];
        imgMat.get(0, 0, imgData);  // Get all pixel data
        // stop timer

        // print shape
//        Log.d("YOLOv8ONNX", "Shape of imgData: " + imgData.length);
        //print contents
//        Log.d("YOLOv8ONNX", "Contents of imgData: " + Arrays.toString(imgData));


        // Rearrange the image data from HWC to CHW format
        for (int c = 2; c >= 0; c--) {
            int offset = 0;
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    // Rearrange from HWC to CHW
                    imageDataBuffer.put(imgData[offset + c]);
                    offset = offset + 3;
//                    imageDataBuffer.put(imgData[index++]);
                }
            }
        }
        // stop timer

//        Log.d("YOLOv8ONNX", "NEW ARRAY: " + imageDataBuffer.get(0));
//        Log.d("YOLOv8ONNX", "NEW ARRAY: " + imageDataBuffer.get(1));
//        Log.d("YOLOv8ONNX", "NEW ARRAY: " + imageDataBuffer.get(2));

        imageDataBuffer.flip();

        return imageDataBuffer;
    }


    // Run inference
    public float[] runInference(Bitmap bitmap) throws OrtException {
        Log.d("YOLOv8ONNX", "RUNNING INFERENCE");
        // Preprocess the image
        long time_preprocess_start = System.currentTimeMillis();
        imageDataBuffer = preprocessImage(bitmap);
        long time_preprocess_end = System.currentTimeMillis();
        Log.d("YOLOv8ONNX", "Preprocessing time: " + (time_preprocess_end - time_preprocess_start) + "ms");

        // Create ONNX tensor
        long[] shape = {1, 3, inputHeight, inputWidth};  // Batch size = 1, Channels = 3, Height, Width
        long time_tensor_start = System.currentTimeMillis();
        inputTensor = OnnxTensor.createTensor(env, imageDataBuffer, shape);
//        float[][][][] tensorData = (float[][][][]) inputTensor.getValue();
        long time_tensor_end = System.currentTimeMillis();
        Log.d("YOLOv8ONNX", "Tensor creation time: " + (time_tensor_end - time_tensor_start) + "ms");

//        Log.d("YOLOv8ONNX", session.getInputInfo().toString());

        // start a timer
        long startTime = System.currentTimeMillis();

        // Run inference
        OrtSession.Result result = session.run(Collections.singletonMap("images", inputTensor));

        //
        long end_time = System.currentTimeMillis();
        Log.d("YOLOv8ONNX", "Inference time: " + (end_time - startTime) + "ms");

        // Get output tensor (depends on your YOLOv8 model's output format)
        Log.d("YOLOv8ONNX", result.get(0).toString());

        float[][][] outputData = (float[][][]) result.get(0).getValue();

        List<PostProcessor.Detection> data_post_processed = processOutput(outputData);
        Log.d("YOLOv8ONNX", "Number of detections: " + data_post_processed.size());
        for (PostProcessor.Detection detection : data_post_processed) {
            Log.d("YOLOv8ONNX", "Bounding box: x=" + detection.getX() + ", y=" + detection.getY() + ", width=" + detection.getWidth() + ", height=" + detection.getHeight());
        }
        // Loop through the output tensor
        for (int gridCell = 0; gridCell < outputData[0][0].length; gridCell++) {
            // First 4 values are bounding box predictions

            float x = outputData[0][0][gridCell];     // x-coordinate
            float y = outputData[0][1][gridCell];     // y-coordinate
            float width = outputData[0][2][gridCell]; // width
            float height = outputData[0][3][gridCell];// height

            float current_max = 0.0f;
            int current_max_index = 0;
            int num_classes = 80;
            for (int class_id = 0; class_id < num_classes; class_id++){
                int current_class_offset = class_id +4;
                    if ((float) outputData[0][current_class_offset][gridCell] > 0.5){
                        System.out.println("Bounding box: x=" + x + ", y=" + y + ", width=" + width + ", height=" + height + ", object confidence: " + outputData[0][current_class_offset][gridCell] + ", Class ID: " + class_id);
                    }
            }
        }

//        System.out.println(outputData[0][4][6]);
//            for (int i = 0; i < outputData[0][6].length; i++) {
//                if (i > 8200 && i < 8215)
//                System.out.println("Index: " + i + ", Prob: " + outputData[0][6][i]);
//            }

        float[] output;
        // Initialize empty arra with values [4.0,4.9]
        output = new float[]{4.0f, 4.9f};


        return output;  // Return the inference output
    }
}
