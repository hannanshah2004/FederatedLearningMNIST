import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Activation
import copy
import sys

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255
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
    new_weights = []
    for weights_list_tuple in zip(*weights):
        new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
    return new_weights

def get_model_size(model):
    """Returns the size of the model's weights in bytes."""
    total_size = 0
    for layer in model.get_weights():
        total_size += sys.getsizeof(layer)
    return total_size

def centralized_training(x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
    model = SimpleMLP().build(784, 10)
    model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    
    data_size = sys.getsizeof(x_train) + sys.getsizeof(y_train)
    centralized_communication = data_size 
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return accuracy, centralized_communication

def federated_learning(clients, x_test, y_test, global_model, comms_rounds=10, epochs=1, batch_size=32):
    total_communication = 0
    accuracies = []
    for comm_round in range(comms_rounds):
        client_weights = []
        for client in clients:
            local_model = copy.deepcopy(global_model)
            local_data = batch_data(clients[client], bs=batch_size)
            local_model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            local_model.fit(local_data, epochs=epochs, verbose=0)
            client_weights.append(local_model.get_weights())
            
            total_communication += get_model_size(local_model)
        
        global_weights = federated_averaging(client_weights)
        global_model.set_weights(global_weights)
        
        total_communication += get_model_size(global_model)
    
    global_model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    accuracies.append(accuracy)
    
    return accuracies[-1], total_communication

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset()

    centralized_acc, centralized_communication = centralized_training(x_train, y_train, x_test, y_test)
    print(f"Centralized Training Accuracy: {centralized_acc * 100:.2f}%")
    print(f"Centralized Communication Overhead: {centralized_communication / (1024 ** 2):.2f} MB")

    clients = create_clients(x_train, y_train, num_clients=20)
    global_model = SimpleMLP().build(784, 10)
    federated_acc, federated_communication = federated_learning(clients, x_test, y_test, global_model)
    print(f"Federated Learning Accuracy: {federated_acc * 100:.2f}%")
    print(f"Federated Communication Overhead: {federated_communication / (1024 ** 2):.2f} MB")

    accuracy_improvement = ((federated_acc - centralized_acc) / centralized_acc) * 100
    print(f"Accuracy Improvement: {accuracy_improvement:.2f}%")

    communication_reduction = ((centralized_communication - federated_communication) / centralized_communication) * 100
    print(f"Communication Overhead Reduction: {communication_reduction:.2f}%")
