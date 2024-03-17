import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Activation

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1 )).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

def create_clients(x_train, y_train, num_clients=10, initial='client'):
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    data = list(zip(x_train, y_train))
    random.shuffle(data)
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    return {client_names[i]: shards[i] for i in range(len(client_names))}

def batch_data(data_shard, bs=32):
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def federated_averaging(weights):
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
    return new_weights

(x_train, y_train), (x_test, y_test) = load_mnist_dataset()

clients = create_clients(x_train, y_train)

global_model = SimpleMLP().build(784, 10)

comms_round = 10

for comm_round in range(comms_round):
    local_weights = []
    global_weights = global_model.get_weights()
    
    for client_name, data in clients.items():
        local_model = SimpleMLP().build(784, 10)
        local_model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
        local_model.set_weights(global_weights)
        
        client_dataset = batch_data(data)
        local_model.fit(client_dataset, epochs=1, verbose=0)
        
        local_weights.append(local_model.get_weights())
        
        tf.keras.backend.clear_session()
    
    global_weights = federated_averaging(local_weights)
    global_model.set_weights(global_weights)

    global_model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
    
    loss, acc = global_model.evaluate(x_test.reshape((-1, 784)), y_test, verbose=0)
    print(f'At round {comm_round+1}: Global Accuracy : {acc*100:.2f}%')

print("We did it! Federated Learning is Completed.")
