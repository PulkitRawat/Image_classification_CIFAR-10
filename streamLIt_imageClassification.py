import streamlit as st
import numpy as np 
import tensorflow as tf
import cv2 as cv
import os 
import tempfile

from tensorflow import keras

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = keras.models.load_model('IC_model')

# considering the image will be given in the same format as the data
def prediction(imagePath):
    img = cv.imread(imagePath)
    img = cv.resize(img, (32,32))
    img = img/255
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    class_index = np.argmax(predictions)
    class_label = class_names[class_index]

    return class_label

st.title('Image Classification App')
st.write('Upload an Image')
uploaded_image = st.file_uploader('Choose an Image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    temp_dir = tempfile.TemporaryDirectory()
    with open(os.path.join(temp_dir.name, uploaded_image.name), 'wb') as f:
        f.write(uploaded_image.read())
    
    # Get the file path of the saved image
    image_path = os.path.join(temp_dir.name, uploaded_image.name)

    if st.button('Predict'):
        predictions = prediction(image_path)
        st.image(uploaded_image, caption= predictions, use_column_width=True)

  