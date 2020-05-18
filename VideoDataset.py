import os
import cv2 as cv
import numpy as np
import skvideo
import torch
#import torchvision
from torch import nn
from skvideo import io as sk

'''
    Class --> Label
        Handstand --> 0
        Smile --> 1
        situp --> 2
        
    
    Clip duration 16 frames
    Height - Width : 20x20
'''

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='hdb', clip_duration = 16):
        self.path = dataset
        self.duration = clip_duration

        self.crop_size = 100
        self.videos = []
        self.classes = os.listdir(self.path)
        for video_class in self.classes:
            videos = os.listdir(self.path + '/' + video_class)
            for video in videos:
                video_dict = {'path' : self.path + '/' + video_class + '/' + video, 'label' : video_class}
                self.videos.append(video_dict)

        self.num_examples = len(self.videos)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        video_data = self.videos[index]
        video_file = video_data['path']
        video_label = video_data['label']
        if video_label == 'smile':
            video_label = 1
        elif video_label == 'handstand':
            video_label = 0
        else:
            video_label = 2
        #sk.vread(video_file)
       
        cap = cv.VideoCapture(video_file)
        video = None
        ret = True
        while(ret):
         ret, frame = cap.read()
         if ret:
          #print('frame')
          #print(frame.shape)
          if video is None:
              video = frame.reshape((1,frame.shape[0],frame.shape[1],frame.shape[2]))
          else:
              video = np.vstack((video, frame.reshape((1,frame.shape[0],frame.shape[1],frame.shape[2]))))

        cap.release()
        #print(video.shape)
        #exit(2)
        time_index = np.random.randint(video.shape[0] - self.duration)
        height_index = np.random.randint(video.shape[1] - self.crop_size)
        width_index = np.random.randint(video.shape[2] - self.crop_size)

        video = video[time_index: time_index + self.duration, height_index:height_index + self.crop_size, width_index: width_index + self.crop_size]
        video = torch.from_numpy(video).permute(3,0,1,2)
        video = video/255.0
        return video, video_label
