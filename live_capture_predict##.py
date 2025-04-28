



import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("model2.h5")
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
        resized = cv2.resize(threshed, (28, 28))
        input_img = resized.reshape(1, 28, 28, 1)
        prediction = model.predict(input_img)
        predicted_label = np.argmax(prediction)
        frame_with_prediction = cv2.putText(frame, str(predicted_label), (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-time prediction', frame_with_prediction)
        # Display the resized image in a different window
        # We upscale it so that it's easier to see
        upscaled_resized = cv2.resize(resized, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Resized Image', upscaled_resized)
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()   
    cv2.destroyAllWindows()