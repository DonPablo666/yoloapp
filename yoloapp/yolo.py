import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors
model_path = 'path_to_your_model.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape details
input_shape = input_details[0]['shape']

# OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Process the detections
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            # Extract bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]

            # Convert coordinates to pixels
            height, width, _ = frame.shape
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            label = f'Class: {int(classes[i])}, Score: {scores[i]:.2f}'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv5 TFLite Object Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
