import os.path as osp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_loaders.custom_scenes import CustomScenes
import numpy as np

def fix_coord_custom_nerf(args, train_set, val_set):
    """
    Apply coordinate transformations for NeRF training ONLY
    
    Transformation pipeline for NeRF:
    1. Center the scene using move_all_cam_vec (NO scaling here)
    2. Scale to NeRF's coordinate system using pose_scale
    3. Apply pose_scale2 for NeRF rendering
    """
    train_poses = train_set.poses
    val_poses = val_set.poses
    all_poses = np.concatenate([train_poses, val_poses])
    
    # Reshape to (N, 3, 4)
    all_poses = all_poses.reshape(-1, 3, 4)
    
    # Step 1: Move to origin (NO scaling applied to move_vec)
    if train_set.move_all_cam_vec != [0., 0., 0.]:
        move_vec = np.array(train_set.move_all_cam_vec)
        all_poses[:, :3, 3] += move_vec
        print(f"[NeRF] Applied move_all_cam_vec: {move_vec}")
    
    # Step 2: Apply pose_scale (scale to NeRF coordinate system)
    sc = train_set.pose_scale
    all_poses[:, :3, 3] *= sc
    print(f"[NeRF] Applied pose_scale: {sc}")
    
    # Step 3: Apply pose_scale2 (additional NeRF-specific scaling)
    if train_set.pose_scale2 != 1.0:
        all_poses[:, :3, 3] *= train_set.pose_scale2
        print(f"[NeRF] Applied pose_scale2: {train_set.pose_scale2}")
    
    bounds = np.array([train_set.near, train_set.far])
    
    # Flatten back to (N, 12)
    all_poses = all_poses.reshape(-1, 12)
    train_set.poses = all_poses[:train_poses.shape[0]]
    val_set.poses = all_poses[train_poses.shape[0]:]
    
    # Store transform parameters
    train_set.transform_params = {
        'pose_scale': train_set.pose_scale,
        'pose_scale2': train_set.pose_scale2,
        'move_all_cam_vec': train_set.move_all_cam_vec
    }
    val_set.transform_params = train_set.transform_params
    print(f"[NeRF] Transform params stored.")
    return train_set, val_set, bounds


def fix_coord_custom_dfnet(args, train_set, val_set):
    """
    Apply coordinate transformations for DFNet training
    
    Transformation pipeline for DFNet:
    1. Center the scene using move_all_cam_vec (NO scaling here)
    2. NO pose_scale applied - DFNet works at original scene scale
    3. NO pose_scale2 applied - that's only for NeRF
    
    DFNet must work at the SAME SCALE as the original scene!
    """
    train_poses = train_set.poses
    val_poses = val_set.poses
    all_poses = np.concatenate([train_poses, val_poses])
    
    # Reshape to (N, 3, 4)
    all_poses = all_poses.reshape(-1, 3, 4)
    
    # ONLY recenter - NO SCALING for DFNet
    if train_set.move_all_cam_vec != [0., 0., 0.]:
        move_vec = np.array(train_set.move_all_cam_vec)
        all_poses[:, :3, 3] += move_vec
        print(f"[DFNet] Applied move_all_cam_vec: {move_vec}")
        print(f"[DFNet] NO scaling applied - keeping original scene scale")
    
    bounds = np.array([train_set.near, train_set.far])
    
    # Flatten back to (N, 12)
    all_poses = all_poses.reshape(-1, 12)
    train_set.poses = all_poses[:train_poses.shape[0]]
    val_set.poses = all_poses[train_poses.shape[0]:]
    
    # Store transform parameters (for inference)
    train_set.transform_params = {
        'move_all_cam_vec': train_set.move_all_cam_vec,
        'pose_scale': 1.0,  # NOT used for DFNet
        'pose_scale2': 1.0  # NOT used for DFNet
    }
    val_set.transform_params = train_set.transform_params
    print(f"[DFNet] Transform params stored without scaling.")
    return train_set, val_set, bounds

def load_custom_dataloader(args):
    """Data loader for Custom dataset - PoseNet/DFNet training"""
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

    if hasattr(args, 'world_setup_path'):
        world_setup_path = args.world_setup_path
    else:
        world_setup_path = None
    
    kwargs = dict(
        data_path=args.datadir,
        transform=data_transform,
        target_transform=target_transform,
        df=args.df,
        ret_idx=ret_idx,
        fix_idx=fix_idx,
        ret_hist=ret_hist,
        hist_bin=args.hist_bin,
        world_setup_path=world_setup_path
    )
    
    train_set = CustomScenes(train=True, trainskip=args.trainskip, **kwargs)
    val_set = CustomScenes(train=False, testskip=args.testskip, **kwargs)
    
    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx
    
    # Use DFNet coordinate transform (NO SCALING)
    train_set, val_set, bounds = fix_coord_custom_dfnet(args, train_set, val_set)
    
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
    
    if hasattr(args, 'world_setup_path'):
        world_setup_path = args.world_setup_path
    else:
        world_setup_path = None
    
    kwargs = dict(
        data_path=args.datadir,
        transform=data_transform,
        target_transform=target_transform,
        df=args.df,
        ret_idx=ret_idx,
        fix_idx=fix_idx,
        ret_hist=ret_hist,
        hist_bin=args.hist_bin,
        world_setup_path=world_setup_path
    )
    
    train_set = CustomScenes(train=True, trainskip=args.trainskip, **kwargs)
    val_set = CustomScenes(train=False, testskip=args.testskip, **kwargs)
    
    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx
    
    # Use NeRF coordinate transform (WITH SCALING)
    train_set, val_set, bounds = fix_coord_custom_nerf(args, train_set, val_set)
    
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