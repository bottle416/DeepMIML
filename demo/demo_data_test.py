# -*- coding: utf-8 -*-

from __future__ import print_function
import skimage.io as io
import sys
sys.path.insert(0, "lib")
from pycocotools.coco import COCO
import numpy as np
import skimage.draw as draw

dataDir='..'
dataType='val2017'
annFile='data/coco/annotations/instances_{}.json'.format(dataType)
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
# 找到符合'person','dog','skateboard'过滤条件的category_id
# 找到符合'person','dog','skateboard'过滤条件的category_id
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# 找出符合category_id过滤条件的image_id
imgIds = coco.getImgIds(catIds=catIds );
# 找出imgIds中images_id为324158的image_id
imgIds = coco.getImgIds(imgIds = [324158])
# 加载图片，获取图片的数字矩阵
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# 显示图片
I = io.imread(img['coco_url'])

# 创建一个与图像大小相同的空白图像
canvas = np.zeros_like(I)

# 在空白图像上绘制原始图像
canvas[:,:,:] = I

# 保存图片
io.imsave('output_image.png', canvas)
