import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, num_samples=100000, width=1024, height=576, sample_frames=2, object_frame=0):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = "/scratch/nua3jz/Datasets/DAVIS_OBJECTS"
        self.gt_base_folder = "/scratch/nua3jz/Datasets/DAVIS/JPEGImages/Full-Resolution"
        self.folders = os.listdir(self.base_folder)
        self.num_samples = num_samples
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.object_frame = object_frame
        

    def __len__(self):
        return self.num_samples
    
    def prepaer_boxes(self, raw_boxes, max_objs=1):
        boxes = torch.zeros(max_objs, 4)
        masks = torch.zeros(max_objs)
         
        for idx in range(raw_boxes.shape[0]):
            boxes[idx] = torch.tensor(raw_boxes[idx])
            masks[idx] = 1
        
        return boxes, masks
        
    def __getitem__(self, idx):
        
        # Randomly select a folder (representing a video) from the base folder
        # chosen_folder = self.folders[idx]
        chosen_folder = random.choice(self.folders)
        
        video_frames_path = os.path.join(self.base_folder, chosen_folder, "frames")
        frames = sorted(os.listdir(video_frames_path))

        object_images_path = os.path.join(self.base_folder, chosen_folder, "object_images")
        obj_images = sorted(os.listdir(object_images_path))

        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")

        # Randomly select a start index for frame sequence
        # start_idx = random.randint(0, len(frames) - self.sample_frames)
        selected_frames = frames[0:0 + self.sample_frames]

        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(video_frames_path, frame_name)
            with Image.open(frame_path) as img:
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1

                # Rearrange channels if necessary
                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(dim=2, keepdim=True)  # For grayscale images

                pixel_values[i] = img_normalized
        
        # choose segmented object images of first frame
        obj_image = Image.open(os.path.join(object_images_path, obj_images[self.object_frame]))
        obj_image_resized = obj_image.resize((self.width, self.height))
        obj_image_tensor = torch.from_numpy(np.array(obj_image_resized)).float()
        obj_image_normalized = obj_image_tensor / 127.5 - 1
        obj_image_normalized = obj_image_normalized.permute(2, 0, 1)  # For RGB images
        
        bboxes_path = os.path.join(self.base_folder, chosen_folder, "bbox.npy")
        bboxes = np.load(bboxes_path)[self.object_frame:self.object_frame+1, :]
        bboxes, masks = self.prepaer_boxes(bboxes)
        
        gt_video_frames_path = os.path.join(self.gt_base_folder, chosen_folder)
        gt_frames = sorted(os.listdir(gt_video_frames_path))
        selected_gt_frames = gt_frames[0:0+self.sample_frames]
        gt_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        # Load and process each frame
        for i, frame_name in enumerate(selected_gt_frames):
            frame_path = os.path.join(gt_video_frames_path, frame_name)
            with Image.open(frame_path) as img:
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1

                # Rearrange channels if necessary
                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(dim=2, keepdim=True)  # For grayscale images

                gt_pixel_values[i] = img_normalized
        
        
        return {
            'pixel_values': pixel_values, 
            'object_image': obj_image_normalized,
            'bboxes': bboxes, 
            'bboxes_masks': masks,
            "gt_pixel_values": gt_pixel_values
            }
