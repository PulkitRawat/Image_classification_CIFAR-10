import pickle 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers, models

def unpickle(file:str):
    with open (file,'rb') as f:
        dict = pickle.load(f, encoding= 'bytes')
    return dict

def preprocessing(data_dict: dict):
    data = np.array(data_dict[b'data'])
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3 ,1)
    data = data/255.0
    labels = np.array(data_dict[b'labels'])
    return data, labels

train_data = []
# print(train.keys())

for i in range(1,6):
    file = f'data_batch_{i}'
    x = unpickle(file)
    train_data.append(x)

train_images_batches = []
train_labels_batches = []

for batch_data in train_data:
    images, labels = preprocessing(batch_data)
    train_images_batches.append(images)
    train_labels_batches.append(labels)

test_data = unpickle('test_batch')

train_images  = np.concatenate(train_images_batches, axis =0)
train_labels = np.concatenate(train_labels_batches, axis =0)
test_images, test_labels = preprocessing(test_data)

# MaxPooling layers are inserted after certain Conv2D layers to reduce spatial dimensions and enhance the network's ability to capture hierarchical features.
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_split= 0.2)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")

# # Plot training history
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()

model.save('IC_model')