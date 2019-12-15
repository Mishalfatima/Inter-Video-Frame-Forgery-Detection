import requests
from tqdm import tqdm
import cv2
import os
import tarfile
import numpy as np
import math
import sys
import torch
from PIL import Image
class Dashcam_data():
    def __init__(self, url='http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz',
                 dataset='HEV1', dir='./data/', batch_size=64, frame_size=[112, 112],
                 train="train", seq=False, mean_file='mean_file.npy'):
        self.url = url
        self.dataset = dataset
        self.dir = dir
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.train = train
        self.mean_file = mean_file
        self.dir_structure = {}
        self.im_names = []
        self.im_pointer = 0
        self.batch = []
        self.aug_steps = [1]
        self.im_pointer = 0

        self.paths = []
        self.labels = []
        self.paths_abnormal =[]
        self.labels_abnormal = []

        if (self.train == "train"):

            self.cat_path = "/hdd/local/sda/mishal/Anticipating-Accidents-master/dataset/videos/training/frames_train"

        else:

            self.cat_path = "/hdd/local/sda/mishal/Anticipating-Accidents-master/dataset/videos/testing/frames/negative"



        for folder in os.listdir(self.cat_path):

                path_abnormal = os.path.join(self.cat_path, folder)
                self.paths.append(path_abnormal)
                self.labels.append(0)

        for folder in os.listdir(self.cat_path):
            path_normal = os.path.join(self.cat_path, folder)
            self.paths.append(path_normal)
            self.labels.append(1)


        print(len(self.paths))
        print(len(self.labels))

        self.total_folders=len(self.paths)
        self.im_ind = list(range(len(self.paths)))



    def get_all_im_names(self,cat_path):
        # print(self.cat_path)
        if os.path.exists('im_names.py'):
            self.im_names = np.load('im_names.npy')
        else:
            for dir, subdir, files in os.walk(cat_path):
                tif_files = [x for x in files if x.endswith('.jpg')]
                if len(tif_files) != 0:
                    seq_file_names = [dir + '/' + x for x in tif_files]
                    seq_file_names = np.array(seq_file_names)
                    # print(np.sort(seq_file_names))
                    if len(self.im_names) != 0:
                        self.im_names = np.concatenate((self.im_names, seq_file_names))
                        # self.im_names = [self.im_names, seq_file_names]
                    else:
                        self.im_names = seq_file_names
            self.im_names = np.sort(self.im_names)

        return self.im_names
        # self.im_names.sort()

    def get_mean_file(self):
        print("Computing the mean of all training frames")
        for im_count in range(len(self.im_names)):
            if im_count == 0:
                im_sum = cv2.imread(self.im_names[im_count])
            else:
                im = cv2.imread(self.im_names[im_count])
                im_sum += im
            if im_count % 100 == 0:
                print("Computed the mean of {} frames".format(im_count))
        self.mean_im = im_sum / (1.0 * len(self.im_names) * 255.0)

        # gray scale conversion
        if np.ndim(self.mean_im) > 2:
            self.mean_im = cv2.cvtColor(self.mean_im.astype(np.float32), cv2.COLOR_RGB2GRAY)

        np.save(self.mean_file, self.mean_im)

    def get_next_batch(self,batch_size,clip_len):

        if (self.im_pointer == 0):
            np.random.shuffle(self.im_ind)

        self.total_folders = len(self.paths)

        self.batch = np.zeros((batch_size, clip_len, self.frame_size[0],
                               self.frame_size[1], 3))

        self.l = np.zeros((batch_size))


        for idx in range(batch_size):


            images = np.zeros((clip_len, self.frame_size[0],
                               self.frame_size[1], 3))
            video = self.paths[self.im_ind[self.im_pointer]]
            label = self.labels[self.im_ind[self.im_pointer]]

            path, dirs, files = next(os.walk(video))


            paths = [video + "/" + x for x in files]

            frames = np.sort(paths)

            num_frames = len(frames)

            time_index = np.random.randint(num_frames - clip_len)

            sequence = frames[time_index:time_index + clip_len]

            if (label == 1):
                time_index_1 = np.random.randint(clip_len)
                time_index_2 = np.random.randint(num_frames-clip_len)


                sequence[time_index_1] = frames[time_index_2]

                #For fixed location

                #sequence[8] = frames[time_index_2]

            for file in range(len(sequence)):
                img=cv2.imread(sequence[file])/255
                img = cv2.resize(img,(self.frame_size[0], self.frame_size[1]))
                images[file,:,:,:] = img

            self.batch[idx,:,:,:,:] = images
            self.l[idx] = label

            self.im_pointer+=1
            if (self.im_pointer==len(self.paths)):
                self.im_pointer = 0
                np.random.shuffle(self.im_ind)

        self.batch = np.moveaxis(self.batch,4,1)



        return torch.from_numpy(self.batch).float(), torch.from_numpy(self.l).long()


if __name__ == '__main__':

    dataset = Dashcam_data("val")

    im_names = (dataset.total_folders)
    tot_batches = int(im_names / 10)

    for i in range(tot_batches):
        print(i, "out of", tot_batches)
        batch,labels = dataset.get_next_batch(10,16)






