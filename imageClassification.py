import pickle 
import numpy as np

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

for i in range(6):
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
tran_labels = np.concatenate(train_labels_batches, axis =0)
test_images, test_labels = preprocessing(test_data)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
