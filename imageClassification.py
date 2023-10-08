import pickle 
import numpy as np

def unpickle(file:str):
    with open (file,'rb') as f:
        dict = pickle.load(f, encoding= 'bytes')
    return dict


train = unpickle('data_batch_1')
# print(train.keys())

# for i in range(6):
#     file = f'data_batch_{i}'
#     x = unpickle(file)
#     train.append(x)

test = unpickle('test_batch')


