package com.example.onnx_java;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.IOException;
import java.nio.FloatBuffer;

import ai.onnxruntime.OrtException;


public class MainActivity extends AppCompatActivity {
    YOLOv8ONNX yolo;
    Bitmap image_data;
    FloatBuffer image_data_buffer;
    float[] preprocessed_image;
    Bitmap preprocess_image_bitmap;

    Button preprocessButton;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);



        preprocessButton = findViewById(R.id.preprocessButton);
        preprocessButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
//                    // run preprocessing
//                    image_data_buffer = yolo.preprocessImage(image_data);
//
//                    // convert imagebuffer to bitmap
//                    preprocess_image_bitmap = yolo.preprocessImageBitmap(image_data);

//                    // set imageview
//                    ImageView imageView = findViewById(R.id.image);
//                    imageView.setImageBitmap(preprocess_image_bitmap);

                    float[] output = yolo.runInference(image_data);
                    Log.d("MainActivity", "Inference successful");
                    Log.d("MainActivity", "Output: " + output.toString());
                } catch (OrtException e) {
                    throw new RuntimeException(e);
                }
            }
        });


        try {
            yolo = new YOLOv8ONNX("yolov8s.onnx", getApplicationContext());
            Log.d("MainActivity", "Model loaded successfully");
        } catch (OrtException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Load the image
        try {
            image_data = yolo.loadImageFromAssets(getApplicationContext(), "car.jpg");
            Log.d("MainActivity", "Image loaded successfully");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Preprocess the image
//        preprocessed_image = yolo.preprocessImage(image_data);
        Log.d("MainActivity", "Image preprocessed successfully");

        // run inference



    }

    @Override
    protected void onResume() {
        super.onResume();

        // Preprocess the image
//        preprocessed_image = yolo.preprocessImage(image_data);

    }



    // make a button press trigger a function

}