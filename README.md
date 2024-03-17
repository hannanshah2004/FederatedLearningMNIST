# FedCraft
Implementing Federated Learning using TensorFlow, with MNIST database for training and testing. At a high level, it involves multiple clients training a local model using their own data, and then utilizing averaged weights across all the client models. The process to writing the code involved the following fundamental steps : 

1. Loading the MNIST dataset
2. Creating a multilayer perceptron, creating the hidden layer with weights and relu / softmax activation functions
3. Creating the clients to foster a decentralized training model
4. Creating batches from the data into shards in order to prevent overfitting
5. Averaging the weights and using that to update the model
6. Using categorical cross-entropy as a loss function for measuring the performance of the model, outputting probability value in between 0 and 1
7. Updating the global model based on the local weights and bias
8. Iterating through a series of rounds to continually improve the model until our pre-set round 10
9. Reaching as close to convergence as possible considering the size of the dataset and the number of rounds used to implement the federated learning model

Overall, this was super interesting and helped me dive deeper into federated learning!
