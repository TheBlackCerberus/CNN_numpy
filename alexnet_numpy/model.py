import numpy as np
import math
import os
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import copy
#from loss import CategoricalCrossEntropy

class BatchGenerator:
    def __init__(self, samples, batch_size=128, root_dir=None):
        self.index = 0
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(samples) / batch_size)
        self.root_dir = root_dir

    def load_samples(csv_file, path):
        data_frame = pd.read_csv(os.path.join(path,csv_file))
        data_frame = data_frame[['Filename', 'Class', 'Label']]
        file_names = list(data_frame.iloc[:,0])
        # Get the labels present in the second column
        labels = list(data_frame.iloc[:,2])
        samples=[]
        for samp,lab in zip(file_names,labels):
            samples.append([samp,lab])
        return samples

    def generate_batch(self):
        """
        Yields the next training batch.
        Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
        """
        num_samples = len(self.samples)
        while True: # Loop forever so the generator never terminates
            samples = shuffle(self.samples)

            # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
            for offset in range(0, num_samples, self.num_batches):
                # Get the samples you'll use in this batch
                batch_samples = samples[offset:offset+self.batch_size]

                # Initialise X_train and y_train arrays for this batch
                images = []
                labels = []

                # For each example
                for batch_sample in batch_samples:
                    # Load image (X) and label (y)
                    img_name = batch_sample[0]
                    label = batch_sample[1]
                    img = cv2.imread(os.path.join(self.root_dir,img_name))

                    # apply any kind of preprocessing

                    # Add example to arrays
                    images.append(img)
                    labels.append(label)

                # Make sure they're numpy arrays (as opposed to lists)
                images = np.array(images)
                labels = np.array(labels)

                # The generator-y part: yield the next training batch
                yield images, labels


class Model:
    def __init__(self, *layers, **kwargs):
        self.layers = layers
        self.batch_size = 0
        self.loss = None
        self.optimizer = None

    def set_loss(self, loss):
        self.loss = loss


    def fit(self, samples, batch_size=128, epochs=50, root_dir=None):
        for epoch_counter in range(epochs):
            print(f"Epoch {epoch_counter + 1}")
            batch = BatchGenerator(samples, batch_size=batch_size, root_dir=root_dir)
            batch_generator = batch.generate_batch()
            iter = 1
            for batch_counter in range(batch.num_batches):
                images_batch, labels_batch = next(batch_generator)
                print(f"Label batch {iter}: {labels_batch}")
                batch_pred = images_batch.copy()
                print(batch_pred.shape)
                print(labels_batch.shape)
                print("-------------------")
                for layer in reversed(self.layers):
                    batch_pred = layer.forward_pass(batch_pred, save_cache=True)
                    print(f"innner {batch_pred.shape}")
                dA = self.loss.grad(labels_batch, batch_pred)
                for layer in self.layers:
                    dA = layer.backward_pass(dA)
                    # do optimazition
                for layer in self.layers:
                    layer.apply_grads()











