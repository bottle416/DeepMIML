# -*- coding: utf-8 -*-

from __future__ import print_function
import os
# 设置环境变量 CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用第二个 GPU

import keras
import theano
import sys
import random
import numpy as np
sys.path.insert(0, "lib")

from theano import function, config, shared, tensor
from keras.models import model_from_json, Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras import backend as K
from theano import config

# 添加 GPU 配置代码
import theano.gpuarray
import theano.tensor as T

# 使用 GPU 后端
theano.gpuarray.use("cuda")

# 检查是否成功配置了 GPU
if theano.gpuarray.pygpu_activated:
    print("Theano is using GPU.")
else:
    print("Theano is not using GPU.")

from cocodemo.fed_iid_coco_dataset import COCODataset
from cocodemo.fed_iid_coco_layer import COCODataLayer
from cocodemo.coco_dataset import TestCOCODataset
from deepmiml.deepmiml import DeepMIML
from deepmiml.utils import save_keras_model, evaluate
from cocodemo.vgg_16 import VGG_16

# 在每一轮训练结束后进行模型参数的平均更新
def average_models(global_model, local_models):
    """
    Args:
    global_model: 全局模型
    local_models: 包含所有客户端的局部模型的列表
    
    Returns:
    updated_global_model: 更新后的全局模型
    """
    # 初始化全局模型参数
    global_weights = global_model.get_weights()
    
    # 遍历所有客户端的局部模型参数，对参数进行累加
    for local_model in local_models:
        local_weights = local_model.get_weights()
        global_weights = [global_weight + local_weight for global_weight, local_weight in zip(global_weights, local_weights)]
    
    # 计算平均值
    avg_weights = [global_weight / len(local_models) for global_weight in global_weights]
    
    # 更新全局模型参数
    updated_global_model = global_model
    updated_global_model.set_weights(avg_weights)
    
    return updated_global_model

if __name__ == "__main__":
    loss = "binary_crossentropy"
    nb_epoch = 1
    batch_size = 32
    L = 80
    K = 20
    model_name = "miml_vgg_16"
    client_type = "iid"

    vgg_model_path = "models/imagenet/vgg/vgg16_weights_th_dim_ordering_th_kernels.h5"
    base_model = VGG_16(vgg_model_path)
    base_model = Sequential(layers=base_model.layers[: -7])
    base_model.add(Convolution2D(512, 1, 1, activation="relu"))
    base_model.add(Dropout(0.5))

    deepmiml = DeepMIML(L=L, K=K, base_model=base_model)

    # 编译 Deep MIML 模型
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
   
    print("Start Training...")
    
    num_clients = 100
    clients_per_round = 1
    local_epochs = 1
    
    # 在每一轮训练开始前复制全局模型给每个客户端
    local_models = [model_from_json(deepmiml.model.to_json()) for _ in range(num_clients)]
    for model in local_models:
        model.set_weights(deepmiml.model.get_weights())
    
    for epoch in range(nb_epoch):
        print("Epoch {}/{}".format(epoch + 1, nb_epoch))
        
        # 存储所有客户端的局部模型
        
        # 随机选择本轮要训练的客户端0
        selected_clients = random.sample(range(num_clients), clients_per_round)
        
        for client_id in selected_clients:
            # 获取该客户端的本地模型
            client_model = local_models[client_id]
            
            # 对所选客户端的数据进行多轮本地训练
            print("Client {} is training".format(client_id))
            dataset = COCODataset("data/coco", client_type, "train", str(client_id))
            client_data = COCODataLayer(dataset, batch_size=batch_size)
            client_samples_per_epoch = client_data.num_images
            
            for local_epoch in range(local_epochs):
                print("Local Epoch {}/{}".format(local_epoch + 1, local_epochs))
                client_model.fit_generator(client_data.generate(),
                                            samples_per_epoch=client_samples_per_epoch,
                                            nb_epoch=1)  # 每个客户端训练1个 epoch
            
            # 存储客户端的局部模型
            local_models[client_id] = client_model
        
        # 对所有客户端的局部模型进行平均得到新的全局模型
        global_model = average_models(deepmiml.model, local_models)
        
        # 每轮训练后保存模型
        save_keras_model(global_model, "outputs/{}/{}_epoch{}.h5".format("iid", model_name, epoch + 1))
        # deepmiml.model.compile(optimizer=sgd, loss=loss, metrics=["accuracy"])


        global_model.compile(optimizer="adadelta", loss="binary_crossentropy")
        # crate data layer
        dataset_test = TestCOCODataset("data/coco", "val", "2017")
        data_layer_test = COCODataLayer(dataset_test, batch_size=batch_size)

        print("Start Predicting...")
        num_images = dataset_test.num_images
        y_pred = np.zeros((num_images, dataset_test.num_classes))
        y_gt = np.zeros((num_images, dataset_test.num_classes))
        for i in range(0, num_images, batch_size):
            if i // batch_size % 10 == 0:
                print("[progress] ({}/{})".format(i, num_images))
            x_val_mini, y_val_mini = data_layer_test.get_data(i, i + batch_size)
            y_pred_mini = global_model.predict(x_val_mini)
            y_pred[i: i + batch_size] = y_pred_mini
            y_gt[i: i + batch_size] = y_val_mini
        evaluate(dataset_test.classes, y_gt, y_pred, threshold_value=0.5)
