import streamlit as st
import numpy as np 
import tensorflow as tf
import cv2 as cv

from tensorflow import keras

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = keras.models.load_model('IC_model')

# considering the image will be given in the same format as the data
def prediction(imagePath):
    img = cv.imread(imagePath)
    img = img.reshape(-1, 3, 32, 32).transpose(0, 2, 3 ,1)
    img = img/255
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    class_index = np.argmax(predictions)
    class_label = class_names[class_index]

    return class_label

st.title('Image Classification App')
st.write('Upload an Image')
uploaded_image = st.file_uploader('Choose an Image')

if uploaded_image != None:
    st.image(uploaded_image, caption='uploaded image', use_column_width=True)

    if st.button('Predict'):
        predictions = prediction(uploaded_image)
        st.image(uploaded_image, caption= predictions, use_column_width=True)