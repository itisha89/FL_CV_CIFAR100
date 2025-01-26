# Federated Learning with TensorFlow Federated: CIFAR-100 Dataset

This repository demonstrates the implementation of Federated Learning (FL) using TensorFlow Federated (TFF) on the CIFAR-100 dataset, which consists of 60,000 color images, each 32x32 pixels, categorized into 100 different classes. This setup aims to train a Convolutional Neural Network (CNN) model on the federated data distributed across multiple clients.

## Key Features:
- **CIFAR-100 Dataset**: A dataset with 100 classes of images, 600 images per class, split into training and testing sets (50,000 for training and 10,000 for testing).
- **Federated Learning Setup**: Implementing Federated Averaging (FedAvg) algorithm using TensorFlow Federated.
- **Preprocessing**: Normalizing and batching images for federated training.
- **Model**: A CNN model built with Keras for image classification.

## Installation

To install the required dependencies for this project, use the following command:

```bash
pip install --quiet --upgrade tensorflow-federated
```

To enable TensorBoard for visualizations, use the following magic command (in Jupyter notebooks):

```bash
%load_ext tensorboard
```

## Dataset Overview

### CIFAR-100
- **Image size**: 32x32 pixels
- **Number of classes**: 100
- **Training images**: 50,000
- **Test images**: 10,000
- **Superclasses**: 100 classes are grouped into 20 "superclasses".

### Sample Visualizations

#### Example Images

To visualize some example images from the dataset:

```python
import matplotlib.pyplot as plt
example_element = next(iter(example_dataset))
plt.imshow(example_element['image'].numpy())
plt.title(f"Label: {example_element['label'].numpy()}")
plt.show()
```

![image](https://github.com/user-attachments/assets/535962dc-9a5a-4404-b805-4b6e424a3ed1)


#### Label Distribution per Client

The distribution of labels on a sample of clients is visualized as follows:

```python
import collections
import matplotlib.pyplot as plt

# Visualize label counts for clients
f = plt.figure(figsize=(12, 7))
f.suptitle('Label Counts for a Sample of Clients')
...
plt.show()
```
![image](https://github.com/user-attachments/assets/32f24bd6-e8cf-43c3-83a1-8bbbdb834a2e)

## Data Preprocessing

We preprocess the CIFAR-100 data by normalizing the images and formatting the data for input into the Federated Learning model:

```python
def preprocess(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.cast(element['image'], tf.float32) / 255.0,  # Normalize to [0, 1]
            y=tf.cast(element['label'], tf.int64)
        )
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_example_dataset = preprocess(example_dataset)
```

## Federated Data

The data is split into federated datasets, one per client:

```python
def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

federated_train_data = make_federated_data(train_cifar, sample_clients)
```

![image](https://github.com/user-attachments/assets/c58ccd6f-b4bd-457c-968f-ed96e5e469f8)

## Federated Model

The model is a simple CNN, implemented with Keras, for image classification. It consists of convolutional layers followed by dense layers:

```python
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(100, activation='softmax')
    ])
    return model
```

### Federated Learning Model Function

```python
def model_fn():
    keras_model = create_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
```

## Federated Averaging Algorithm

We use the **Federated Averaging (FedAvg)** algorithm to train the model across multiple clients:

```python
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
)
```

## Running the Code

1. **Install Dependencies**: Install TensorFlow and TensorFlow Federated.
2. **Prepare the Data**: Load the CIFAR-100 dataset and preprocess it.
3. **Define the Model**: Create the CNN model using Keras.
4. **Setup Federated Learning**: Split the dataset into federated data, and define the federated learning setup.
5. **Train the Model**: Run the federated learning algorithm using the `FedAvg` optimizer.

## Visualizations

You can visualize the training process and evaluate the model using TensorBoard:

```bash
%tensorboard --logdir logs
```

![image](https://github.com/user-attachments/assets/82edef94-041b-4f41-9dd1-35abf5440df5)

![image](https://github.com/user-attachments/assets/76155478-6ae4-45ca-9255-63a3ae5eb9d3)

![image](https://github.com/user-attachments/assets/f9ce52df-ee88-4c43-b246-c933c716a674)

![image](https://github.com/user-attachments/assets/1176e644-0aa8-4271-ba0a-c83e15f0d62c)



## Conclusion

This setup demonstrates the basic principles of federated learning with TensorFlow Federated, using the CIFAR-100 dataset for a multi-client setup. You can extend this implementation for more complex use cases or real-world data distributions.
