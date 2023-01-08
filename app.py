import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
import streamlit as st

model = load_model("facetracker.h5")

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size
classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

model.compile(opt, classloss, regressloss)

def face_detect(img):
    #img = img[50:500, 50:500,:]
    #height, width, _ = img.shape
    # Crop the image to a size that is half the width and height of the image
    #img = img[0:height//2, 0:width//2,:]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = model.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        top_left = tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int))
        bottom_right = tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int))
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

        # Controls the label rectangle
        label_top_left = tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30]))
        label_bottom_right = tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0]))
        cv2.rectangle(img, label_top_left, label_bottom_right, (255, 0, 0), -1)

        # Controls the text rendered
        text_bottom_left = tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5]))
        cv2.putText(img, 'face', text_bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return img

def main():
    st.title('Face Detection')

    # Allow user to upload image
    image = st.file_uploader("Choose an image:")

    if st.button("Detect"):
        image_np = np.frombuffer(image.read(), np.uint8)
        # Convert the NumPy array to an image
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        detect = face_detect(image_cv)
        # Convert the image back to a NumPy array   
        image_detected_np = cv2.imencode('.jpg', detect)[1]
        # Convert the NumPy array to a bytes object
        image_detected_bytes = image_detected_np.tobytes()
        # Display the image on the webpage
        st.image(image_detected_bytes, width=600)

if __name__ == '__main__':
    main()
