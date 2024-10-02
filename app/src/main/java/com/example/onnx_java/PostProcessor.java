package com.example.onnx_java;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PostProcessor {

    private static final float CONFIDENCE_THRESHOLD = 0.5f; // Adjust as needed
    private static final float NMS_THRESHOLD = 0.4f; // Adjust as needed

    public static List<Detection> processOutput(float[][][] outputData) {
        List<Detection> detections = new ArrayList<>();

        // Assuming outputData[0][0] is x, outputData[0][1] is y, etc.
        for (int gridCell = 0; gridCell < outputData[0][0].length; gridCell++) {
            // Extract bounding box and other details
            float x = outputData[0][0][gridCell];
            float y = outputData[0][1][gridCell];
            float width = outputData[0][2][gridCell];
            float height = outputData[0][3][gridCell];
            float objectConfidence = outputData[0][4][gridCell];
            float[] classProbs = Arrays.copyOfRange(outputData[0][5], gridCell, gridCell + 79);
//            float[] classProbs = Arrays.copyOfRange(outputData[0][5], gridCell * 80, (gridCell + 1) * 80); // Example

            // Apply confidence threshold
            if (objectConfidence < CONFIDENCE_THRESHOLD) continue;

            // Calculate class with highest probability
            int classId = getClassWithMaxProbability(classProbs);
            float classProbability = classProbs[classId];

            // Apply confidence threshold for class probability
            if (classProbability < CONFIDENCE_THRESHOLD) continue;

            // Create a Detection object and add to list
            Detection detection = new Detection(x, y, width, height, objectConfidence * classProbability, classId);
            detections.add(detection);
        }

        // Apply Non-Maximum Suppression (NMS)
        return nonMaxSuppression(detections, NMS_THRESHOLD);
    }

    private static int getClassWithMaxProbability(float[] classProbs) {
        int maxIndex = 0;
        float maxProb = classProbs[0];
        for (int i = 1; i < classProbs.length; i++) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static List<Detection> nonMaxSuppression(List<Detection> detections, float threshold) {
        List<Detection> result = new ArrayList<>();
        detections.sort((d1, d2) -> Float.compare(d2.getScore(), d1.getScore())); // Sort by score descending

        boolean[] suppressed = new boolean[detections.size()];
        Arrays.fill(suppressed, false);

        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;

            Detection d1 = detections.get(i);
            result.add(d1);

            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;

                Detection d2 = detections.get(j);
                if (iou(d1, d2) > threshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    private static float iou(Detection d1, Detection d2) {
        float x1 = d1.getX();
        float y1 = d1.getY();
        float w1 = d1.getWidth();
        float h1 = d1.getHeight();
        float x2 = d2.getX();
        float y2 = d2.getY();
        float w2 = d2.getWidth();
        float h2 = d2.getHeight();

        float interX1 = Math.max(x1, x2);
        float interY1 = Math.max(y1, y2);
        float interX2 = Math.min(x1 + w1, x2 + w2);
        float interY2 = Math.min(y1 + h1, y2 + h2);

        float interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        float area1 = w1 * h1;
        float area2 = w2 * h2;

        return interArea / (area1 + area2 - interArea);
    }

    // Detection class to hold bounding box information and other details
    public static class Detection {
        private final float x;
        private final float y;
        private final float width;
        private final float height;
        private final float score;
        private final int classId;

        public Detection(float x, float y, float width, float height, float score, int classId) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.score = score;
            this.classId = classId;
        }

        public float getX() { return x; }
        public float getY() { return y; }
        public float getWidth() { return width; }
        public float getHeight() { return height; }
        public float getScore() { return score; }
        public int getClassId() { return classId; }
    }
}
