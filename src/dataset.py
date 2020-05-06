import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import random
import os
import cv2
import collections

class FaceDetectionDataset(Dataset):
    """Face dataset."""
    def __init__(self, data_path, transform=None):
        self.cache =  h5py.File(data_path, 'r')
        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:].astype(np.float32)
        self.transform = transform

        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]
        self.train_inds = np.random.permutation(np.arange(n_train_samples))
        self.pos_train_inds = self.train_inds[ self.labels[self.train_inds, 0] == 1.0 ]
        self.neg_train_inds = self.train_inds[ self.labels[self.train_inds, 0] != 1.0 ]
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image[:,:,::-1].copy())
        sample = {'image': image,
                  'label': label}

        return sample


    def get_train_size(self):
        return self.train_inds.shape[0]

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size()//factor//batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
        if only_faces:
            selected_inds = np.random.choice(self.pos_train_inds, size=n, replace=False, p=p_pos)
        else:
            selected_pos_inds = np.random.choice(self.pos_train_inds, size=n//2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(self.neg_train_inds, size=n//2, replace=False, p=p_neg)
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        samples = [self.__getitem__(idx) for idx in sorted_inds]
        batch = {
            'image' : torch.stack([sample['image'] for sample in samples]),
            'label' : torch.stack([torch.tensor(sample['label']) for sample in samples])
        }
        return batch

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[:10*n:10]]
        return (self.images[most_prob_inds,...]).astype(np.float32)
    
    def get_n_least_prob_faces(self, prob, n):
        idx = np.argsort(prob)
        most_prob_inds = self.pos_train_inds[idx[:10*n:10]]
        return (self.images[most_prob_inds,...]).astype(np.float32)

    def get_all_train_faces(self):
        samples = [self.__getitem__(idx) for idx in self.pos_train_inds]
        return torch.stack([sample['image'] for sample in samples])


class PPBdataset:
    def __init__(self, path, skip=1):
        ppb_anno = os.path.join(path,'PPB-2017-metadata.csv')
        image_dir = os.path.join(path, "imgs")
        anno_dict = {}
        with open(ppb_anno) as f:
            for line in f.read().split('\n'):
                ind, name, gender, numeric, skin, country = line.split(',')
                anno_dict[name] = (gender.lower(),skin.lower())

        self.image_files = sorted(os.listdir(image_dir))[::skip] #sample every 4 images for computation time in the lab
        self.raw_images = {
            'male_darker':[],
            'male_lighter':[],
            'female_darker':[],
            'female_lighter':[],
        }

        for filename in self.image_files:
            if not filename.endswith(".jpg"):
                continue
            image = cv2.imread(os.path.join(image_dir,filename))[:,:,::-1]
            gender, skin = anno_dict[filename]
            self.raw_images[gender+'_'+skin].append(image)

    def __get_key(self, gender, skin_color):
        gender = gender.lower()
        skin_color = skin_color.lower()
        assert gender in ['male', 'female']
        assert skin_color in ['lighter', 'darker']
        return '{}_{}'.format(gender, skin_color)
        
    def get_sample_faces_from_demographic(self, gender, skin_color, num=1):
        data = []
        key = self.__get_key(gender, skin_color)
        choices = random.choices(self.raw_images[key], k = num)
        #print(choices)
        min_height = min([im.shape[1] for im in choices])
        
        for im in choices:
            data.append(
                cv2.resize(im, 
                        dsize=(int(im.shape[1] * (min_height/float(im.shape[0]))), min_height),
                        interpolation=cv2.INTER_LINEAR) 
                )
            
        return data

    





