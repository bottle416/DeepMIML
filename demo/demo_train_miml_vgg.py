
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
import sys
sys.path.insert(0, "lib")

from cocodemo.coco_dataset import COCODataset
from cocodemo.coco_data_layer import COCODataLayer
from deepmiml.deepmiml import DeepMIML
from deepmiml.utils import save_keras_model
from cocodemo.vgg_16 import VGG_16


if __name__ == "__main__":
    loss = "binary_crossentropy"
    nb_epoch = 10
    batch_size = 32
    L = 80
    K = 20
    model_name = "miml_vgg_16"

    # crate data layer
    dataset = COCODataset("data/coco", "train", "2017")
    data_layer = COCODataLayer(dataset, batch_size=batch_size)

    vgg_model_path = "models/imagenet/vgg/vgg16_weights_th_dim_ordering_th_kernels.h5"
    #  vgg_model_path = "models/imagenet/vgg/vgg16_weights.h5"
    base_model = VGG_16(vgg_model_path)
    base_model = Sequential(layers=base_model.layers[: -7])
    base_model.add(Convolution2D(512, 1, 1, activation="relu"))
    base_model.add(Dropout(0.5))

    deepmiml = DeepMIML(L=L, K=K, base_model=base_model)
    # deepmiml.model.summary()

    print("Compiling Deep MIML Model...")
    deepmiml.model.compile(optimizer="adadelta", loss=loss, metrics=["accuracy"])

    print("Start Training...")
    samples_per_epoch = data_layer.num_images
    deepmiml.model.fit_generator(data_layer.generate(),
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch)

    save_keras_model(deepmiml.model, "outputs/{}/{}".format(dataset.name, model_name))
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
import sys
import random  # Importing random module for client selection
sys.path.insert(0, "lib")

from cocodemo.coco_dataset import COCODataset
from cocodemo.coco_data_layer import COCODataLayer
from deepmiml.deepmiml import DeepMIML
from deepmiml.utils import save_keras_model
from cocodemo.vgg_16 import VGG_16

from keras.optimizers import SGD

if __name__ == "__main__":
    loss = "binary_crossentropy"
    nb_epoch = 10
    batch_size = 32
    L = 80
    K = 20
    model_name = "miml_vgg_16"

    # Create data layer
    dataset = COCODataset("data/coco", "train", "2017")
    data_layer = COCODataLayer(dataset, batch_size=batch_size)

    vgg_model_path = "models/imagenet/vgg/vgg16_weights_th_dim_ordering_th_kernels.h5"
    base_model = VGG_16(vgg_model_path)
    base_model = Sequential(layers=base_model.layers[: -7])
    base_model.add(Convolution2D(512, 1, 1, activation="relu"))
    base_model.add(Dropout(0.5))

    deepmiml = DeepMIML(L=L, K=K, base_model=base_model)

    # Compiling Deep MIML Model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    deepmiml.model.compile(optimizer=sgd, loss=loss, metrics=["accuracy"])

    print("Start Training...")
    samples_per_epoch = data_layer.num_images
    
    num_clients = 100
    clients_per_round = 10
    
    for epoch in range(nb_epoch):
        print("Epoch {}/{}".format(epoch + 1, nb_epoch))
        
        # Randomly select clients for this round
        selected_clients = random.sample(range(num_clients), clients_per_round)
        
        client_weights = []  # List to store weights from selected clients
        
        for client_id in selected_clients:
            # Train the model on data from selected client
            print("client {} is training".format(client_id))
            client_data = COCODataLayer(dataset, batch_size=batch_size, client_id=client_id)
            client_samples_per_epoch = client_data.num_images_per_client
            deepmiml.model.fit_generator(client_data.generate(),
                                          samples_per_epoch=client_samples_per_epoch,
                                          nb_epoch=1)  # Training for 1 epoch per client
            # Save model weights from the client
            client_weights.append(deepmiml.model.get_weights())
        
        # Average weights from all selected clients
        avg_weights = []
        for weights_list in zip(*client_weights):
            avg_weights.append(np.mean(weights_list, axis=0))
        
        # Update model with average weights
        deepmiml.model.set_weights(avg_weights)
        
        # Evaluate model performance after each epoch
        evaluation_loss, evaluation_accuracy = deepmiml.model.evaluate_generator(data_layer.generate(),
                                                                                 steps=samples_per_epoch // batch_size)
        print("Evaluation - Loss: {:.4f}, Accuracy: {:.4f}".format(evaluation_loss, evaluation_accuracy))
        
        # After training on selected clients and evaluating the model, you may perform model aggregation (Federated Averaging) here
        # For simplicity, let's just save the model after each epoch
        save_keras_model(deepmiml.model, "outputs/{}/{}_epoch{}.h5".format(dataset.name, model_name, epoch + 1))

