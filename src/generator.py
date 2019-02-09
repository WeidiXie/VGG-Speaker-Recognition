# System
import keras
import numpy as np
import utils as ut

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, dim, mp_pooler, augmentation=True, batch_size=32, nfft=512, spec_len=250,
                 win_length=400, sampling_rate=16000, hop_length=160, n_classes=5994, shuffle=True, normalize=True):
        'Initialization'
        self.dim = dim
        self.nfft = nfft
        self.sr = sampling_rate
        self.spec_len = spec_len
        self.normalize =normalize
        self.mp_pooler = mp_pooler
        self.win_length = win_length
        self.hop_length = hop_length


        self.labels = labels
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation_mp(list_IDs_temp, indexes)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation_mp(self, list_IDs_temp, indexes):
        X = [self.mp_pooler.apply_async(ut.load_data,
                                        args=(ID, self.win_length, self.sr, self.hop_length,
                                        self.nfft, self.spec_len)) for ID in list_IDs_temp]
        X = np.expand_dims(np.array([p.get() for p in X]), -1)
        y = self.labels[indexes]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


    def __data_generation(self, list_IDs_temp, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,) + self.dim)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :, :, 0] = ut.load_data(ID, win_length=self.win_length, sr=self.sr, hop_length=self.hop_length,
                                         n_fft=self.nfft, spec_len=self.spec_len)
            # Store class
            y[i] = self.labels[indexes[i]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def OHEM_generator(model, datagen, steps, propose_time, batch_size, dims, nclass):
    # propose_time : number of candidate batches.
    # prop : the number of hard batches for training.
    step = 0
    interval = np.array([i*(batch_size // propose_time) for i in range(propose_time)] + [batch_size])

    while True:
        if step == 0 or step > steps - propose_time:
            step = 0
            datagen.on_epoch_end()

        # propose samples,
        samples = np.empty((batch_size,) + dims)
        targets = np.empty((batch_size, nclass))

        for i in range(propose_time):
            x_data, y_data = datagen.__getitem__(index=step+i)
            preds = model.predict(x_data, batch_size=batch_size)   # prediction score
            errs = np.sum(y_data * preds, -1)
            err_sort = np.argsort(errs)

            indices = err_sort[:(interval[i+1]-interval[i])]
            samples[interval[i]:interval[i+1]] = x_data[indices]
            targets[interval[i]:interval[i+1]] = y_data[indices]

        step += propose_time
        yield samples, targets
