import os.path as osp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_loaders.custom_scenes import CustomScenes
import numpy as np

def fix_coord_custom(args, train_set, val_set):
    """
    Apply coordinate transformations if needed
    For now, we'll use the poses as-is from the JSON
    """
    train_poses = train_set.poses
    val_poses = val_set.poses
    all_poses = np.concatenate([train_poses, val_poses])
    
    # You can add coordinate system transformations here if needed
    # For example, converting from nerfstudio to DFNet conventions
    
    bounds = np.array([train_set.near, train_set.far])
    
    # Apply any scaling if needed
    if hasattr(args, 'pose_scale') and args.pose_scale != 1.0:
        all_poses = all_poses.reshape(-1, 3, 4)
        all_poses[:, :3, 3] *= args.pose_scale
        all_poses = all_poses.reshape(-1, 12)
    
    train_set.poses = all_poses[:train_poses.shape[0]]
    val_set.poses = all_poses[train_poses.shape[0]:]
    
    return train_set, val_set, bounds

def load_custom_dataloader(args):
    """Data loader for Custom dataset - PoseNet training"""
    if not args.pose_only:
        raise Exception('load_custom_dataloader() currently only supports PoseNet Training')
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
    
    ret_idx = False
    fix_idx = False
    ret_hist = False
    
    if hasattr(args, 'NeRFH') and args.NeRFH:
        ret_idx = True
        if hasattr(args, 'fix_index') and args.fix_index:
            fix_idx = True
    
    if hasattr(args, 'encode_hist') and args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True
    
    kwargs = dict(
        data_path=args.datadir,
        transform=data_transform,
        target_transform=target_transform,
        df=args.df,
        ret_idx=ret_idx,
        fix_idx=fix_idx,
        ret_hist=ret_hist,
        hist_bin=args.hist_bin
    )
    
    train_set = CustomScenes(train=True, trainskip=args.trainskip, **kwargs)
    val_set = CustomScenes(train=False, testskip=args.testskip, **kwargs)
    
    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx
    
    train_set, val_set, bounds = fix_coord_custom(args, train_set, val_set)
    
    train_shuffle = True
    if hasattr(args, 'eval') and args.eval:
        train_shuffle = False
    
    train_dl = DataLoader(train_set, batch_size=args.batch_size, 
                         shuffle=train_shuffle, num_workers=8)
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, 
                       shuffle=False, num_workers=2)
    test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    
    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]
    
    return train_dl, val_dl, test_dl, hwf, i_split, bounds.min(), bounds.max()

def load_custom_dataloader_NeRF(args):
    """Data loader for Custom dataset - NeRF training"""
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
    
    ret_idx = False
    fix_idx = False
    ret_hist = False
    
    if hasattr(args, 'NeRFH') and args.NeRFH:
        ret_idx = True
        if hasattr(args, 'fix_index') and args.fix_index:
            fix_idx = True
    
    if hasattr(args, 'encode_hist') and args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True
    
    kwargs = dict(
        data_path=args.datadir,
        transform=data_transform,
        target_transform=target_transform,
        df=args.df,
        ret_idx=ret_idx,
        fix_idx=fix_idx,
        ret_hist=ret_hist,
        hist_bin=args.hist_bin
    )
    
    train_set = CustomScenes(train=True, trainskip=args.trainskip, **kwargs)
    val_set = CustomScenes(train=False, testskip=args.testskip, **kwargs)
    
    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx
    
    train_set, val_set, bounds = fix_coord_custom(args, train_set, val_set)
    
    train_shuffle = True
    if (hasattr(args, 'render_video_train') and args.render_video_train) or \
       (hasattr(args, 'render_test') and args.render_test):
        train_shuffle = False
    
    train_dl = DataLoader(train_set, batch_size=1, shuffle=train_shuffle)
    val_dl = DataLoader(val_set, batch_size=1, shuffle=False)
    
    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]
    
    render_poses = None
    render_img = None
    
    return train_dl, val_dl, hwf, i_split, bounds, render_poses, render_img