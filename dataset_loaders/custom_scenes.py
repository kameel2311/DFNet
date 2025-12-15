import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import cv2
import json
from dataset_loaders.utils.color import rgb_to_yuv

def load_image(filename):
    try:
        img = Image.open(filename)
        img = img.convert('RGB')
        return img
    except Exception as e:
        print(f'Could not load image {filename}, Error: {e}')
        return None

class CustomScenes(data.Dataset):
    def __init__(self, data_path, train, transform=None,
                 target_transform=None, seed=7, df=1., 
                 trainskip=1, testskip=1, train_split=0.9,
                 ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10, all_images=False,
                 world_setup_path=None):
        """
        Custom dataset loader for nerfstudio-style JSON format
        
        Args:
            data_path: path to the dataset root
            train: if True, return training images
            transform: transform to apply to images
            target_transform: transform to apply to poses
            df: downscale factor for images
            trainskip: load 1/trainskip images from train set
            testskip: load 1/testskip images from test set
            train_split: ratio of images to use for training
        """
        self.transform = transform
        self.target_transform = target_transform
        self.df = df
        self.train = train
        self.ret_idx = ret_idx
        self.fix_idx = fix_idx
        self.ret_hist = ret_hist
        self.hist_bin = hist_bin
        self.world_setup_path = world_setup_path
        print(f"CustomScenes: world_setup_path = {self.world_setup_path}")
        np.random.seed(seed)
        
        # Load JSON file
        json_path = osp.join(data_path, 'transforms.json')
        with open(json_path, 'r') as f:
            meta = json.load(f)

        # Load world_setup.json if it exists
        if osp.exists(self.world_setup_path):
            with open(self.world_setup_path, 'r') as f:
                world_setup = json.load(f)
            self.near = world_setup.get('near', 0.1)
            self.far = world_setup.get('far', 10.0)
            self.pose_scale = world_setup.get('pose_scale', 1.0)
            self.pose_scale2 = world_setup.get('pose_scale2', 1.0)
            self.move_all_cam_vec = world_setup.get('move_all_cam_vec', [0.0, 0.0, 0.0])
            print(f"Loaded world_setup.json:")
            print(f"  near={self.near}, far={self.far}")
            print(f"  pose_scale={self.pose_scale}, pose_scale2={self.pose_scale2}")
            print(f"  move_all_cam_vec={self.move_all_cam_vec}")
        else:
            print(f"Warning: world_setup.json not found at {world_setup_path}")
            print("Using default values. This may cause poor results!")
            self.near = 0.0
            self.far = 10.0
            self.pose_scale = 1.0
            self.pose_scale2 = 1.0
            self.move_all_cam_vec = [0.0, 0.0, 0.0]

        # Print dataset info
        print(f"Dataset loaded from {data_path}")
        print(f"Number of frames in dataset: {len(meta['frames'])}")
        print("Near/Far bounds set to: ", self.near, self.far)
        print("Pose scale factors set to: ", self.pose_scale, self.pose_scale2)
        print("Camera movement vector set to: ", self.move_all_cam_vec)        
        
        # Extract camera intrinsics
        self.W = meta['w']
        self.H = meta['h']
        self.focal = meta['fl_x']  # Assuming fl_x == fl_y
        
        # Apply downscale factor
        if self.df != 1.:
            self.H = int(self.H // self.df)
            self.W = int(self.W // self.df)
            self.focal = self.focal / self.df
        
        # Extract frames
        frames = meta['frames']
        n_frames = len(frames)
        
        # Split into train/test
        indices = np.arange(n_frames)
        np.random.shuffle(indices)
        n_train = int(n_frames * train_split)

        if not all_images:
            if train:
                indices = indices[:n_train][::trainskip]
            else:
                indices = indices[n_train:][::testskip]
        else:
            indices = indices

        self.gt_idx = indices
        
        # Extract image paths and poses
        self.rgb_files = []
        self.poses = []
        
        for idx in indices:
            frame = frames[idx]
            
            # Get image path
            img_path = frame['file_path']
            if not osp.isabs(img_path):
                img_path = osp.join(data_path, img_path)
            self.rgb_files.append(img_path)
            
            # Get pose (transform_matrix is 4x4)
            transform_matrix = np.array(frame['transform_matrix'])
            # Convert to 3x4 and flatten to (12,)
            pose = transform_matrix[:3, :4].reshape(-1)
            self.poses.append(pose)
        
        self.poses = np.array(self.poses)
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, index):
        img = load_image(self.rgb_files[index])
        pose = self.poses[index]
        
        if self.df != 1.:
            img_np = (np.array(img) / 255.).astype(np.float32)
            dims = (self.W, self.H)
            img = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        
        if self.ret_idx:
            if self.train and self.fix_idx == False:
                return img, pose, index
            else:
                return img, pose, 0
        
        if self.ret_hist:
            yuv = rgb_to_yuv(img)
            y_img = yuv[0]
            hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.)
            hist = hist / (hist.sum()) * 100
            hist = torch.round(hist)
            return img, pose, hist
        
        return img, pose