import requests
import cv2
import os
import tarfile
import numpy as np
import torch
from PIL import Image
class Dashcam_data():
    def __init__(self,
                 dataset='Dashcam', dir='./data/', batch_size=64, frame_size=[112, 112],
                 train="train", mean_file='mean_file.npy'):

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






