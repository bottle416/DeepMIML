# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import shutil
import sys
sys.path.insert(0, "lib")
from pycocotools.coco import COCO


def count_images_per_category(annFile):
    # Load COCO annotations
    coco = COCO(annFile)

    # Get all category names
    cats = coco.loadCats(coco.getCatIds())
    category_names = [cat['name'] for cat in cats]

    # Initialize counters for each category
    category_counts = {cat_name: 0 for cat_name in category_names}

    # Count images per category
    for cat_name in category_names:
        catIds = coco.getCatIds(catNms=[cat_name])
        imgIds = coco.getImgIds(catIds=catIds)
        category_counts[cat_name] = len(imgIds)

    return category_counts


# 设置参数
annFile = 'data/coco/annotations/instances_train2017.json'

# 计算总数据集每个类别的图像总数
total_images_per_category = count_images_per_category(annFile)

# 输出每个类别的图像总数
print("总数据集每个类别的图像总数:")
for cat_name, total_count in total_images_per_category.items():
    print("    Category {}: {}".format(cat_name, total_count))

# 计算总数量
total_images = sum(total_images_per_category.values())

# 输出总数量
print("总数量:", total_images)
