import cv2
import numpy as np
import random

def im_to_blob(im):
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    blob = im.transpose((2, 0, 1))
    return blob


class COCODataLayer(object):
    def __init__(self, dataset, batch_size=32, client_id=None, num_images_per_client=100):
        """
        Attibutes:
            dataset (COCODataset):
                dataset from which to load the data.
            batch_size (int):
                how many samples per batch to load
            client_id (int):
                ID of the client, used to select different datasets
            num_images_per_client (int):
                number of images to select for each client
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.client_id = client_id
        self.num_images_per_client = num_images_per_client

        self.num_images = self.dataset.num_images
        self.data_y = self.dataset.gt_bag_labels()
        self.cur = 0

        # Get a list of all image indices
        self.image_indices = list(range(self.num_images))
        
        # Randomly select num_images_per_client image indices for the current client
        random.seed(client_id)  # Set random seed based on client_id for reproducibility
        random.shuffle(self.image_indices)
        self.client_image_indices = self.image_indices[:num_images_per_client]

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        random.shuffle(self.image_indices)
        self.cur = 0

    def _get_next_minibatch_inds(self):
        """Return the image indices for the next minibatch."""
        if self.cur + self.batch_size >= len(self.client_image_indices):
            self._shuffle_roidb_inds()
        db_inds = self.client_image_indices[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        return db_inds

    def _get_next_minibatch(self):
        """
        Retrun:
            data_x: num_bag x 3(c) x 224(h) x 224(w)
            data_y: num_bag x 80
        """
        data_x = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float32)
        data_y = np.zeros((self.batch_size, self.dataset.num_classes), dtype=np.float32)
        db_inds = self._get_next_minibatch_inds()
        for i, db_ind in enumerate(db_inds):
            im = self.dataset.image_at(db_ind)
            blob = im_to_blob(im)
            data_x[i] = blob
            data_y[i] = self.data_y[db_ind]
        return data_x, data_y

    def generate(self):
        """
        This function is used for keras.Model.fit_generator
        """
        self.cur = 0
        while True:
            x, y = self._get_next_minibatch()
            yield x, y


if __name__ == '__main__':
    from cocodemo.coco_dataset import COCODataset
    coco = COCODataset('data/coco', 'train', '2017')
    data_layer = COCODataLayer(coco)
    (x_train, y_train) = data_layer.generate().next()
    print(x_train.shape)
    print(y_train.shape)
