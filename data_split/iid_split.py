# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import shutil
import sys
sys.path.insert(0, "lib")
from pycocotools.coco import COCO


def distribute_images_by_categories(annFile, num_clients):
    # Load COCO annotations
    coco = COCO(annFile)

    # Get all category names
    cats = coco.loadCats(coco.getCatIds())
    category_names = [cat['name'] for cat in cats]

    # Create directories for each client and initialize counters
    output_dir_base = "data/coco/iid"
    try:
        os.makedirs(output_dir_base)
    except OSError:
        pass
    output_dirs = [os.path.join(output_dir_base, "client_{}".format(i)) for i in range(num_clients)]
    for dir_path in output_dirs:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    # Initialize counters for each client and category
    client_category_counts = {client_id: {cat_name: 0 for cat_name in category_names} for client_id in range(num_clients)}

    # Distribute images by categories to each client and count
    print("\nData Distribution to Clients:")
    for cat_name in category_names:
        catIds = coco.getCatIds(catNms=[cat_name])
        imgIds = coco.getImgIds(catIds=catIds)

        avg_img_per_client = len(imgIds) // num_clients
        imgIds_per_client = [imgIds[i:i + avg_img_per_client] for i in range(0, avg_img_per_client * num_clients, avg_img_per_client)]

        for client_id, img_ids in enumerate(imgIds_per_client):
            output_folder = output_dirs[client_id]
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join('data', 'coco', 'train2017', img_info['file_name'])
                shutil.copy(img_path, output_folder)
                client_category_counts[client_id][cat_name] += 1

    # Output the number of images per category for each client
    print("\nNumber of images per category for each client:")
    for client_id, counts in client_category_counts.items():
        print("Client {}: ".format(client_id))
        for cat_name, count in counts.items():
            print("    Category {}: {}".format(cat_name, count))

    # Calculate total number of images for each client
    total_images_per_client = {client_id: sum(counts.values()) for client_id, counts in client_category_counts.items()}
    print("\nTotal number of images for each client:")
    for client_id, total_count in total_images_per_client.items():
        print("    Client {}: {}".format(client_id, total_count))


# 设置参数
annFile = 'data/coco/annotations/instances_train2017.json'
num_clients = 100

# 分发图像
distribute_images_by_categories(annFile, num_clients)
