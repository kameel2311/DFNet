import sys
sys.path.append('../')
import torch
import numpy as np
import os
import os.path as osp
import json
from torchvision import transforms
from tqdm import tqdm
import argparse
import pytorch3d.transforms as transforms3d

from feature.dfnet import DFNet, DFNet_s
from dataset_loaders.custom_scenes import CustomScenes
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def quaternion_angular_error(q1, q2):
    """
    Calculate angular error between two quaternions in degrees
    Args:
        q1, q2: quaternions as torch tensors (4,)
    Returns:
        angular error in degrees
    """
    q1 = q1 / torch.linalg.norm(q1)
    q2 = q2 / torch.linalg.norm(q2)
    d = torch.abs(torch.sum(q1 * q2))
    d = torch.clamp(d, -1., 1.)
    theta = 2 * torch.acos(d) * 180 / np.pi
    return theta.item()


def translation_error(t1, t2):
    """
    Calculate Euclidean distance between two translation vectors
    Args:
        t1, t2: translation vectors as torch tensors (3,)
    Returns:
        translation error in meters
    """
    return torch.linalg.norm(t1 - t2).item()


def pose_matrix_to_quaternion_translation(pose):
    """
    Convert 3x4 pose matrix to quaternion and translation
    Args:
        pose: numpy array (3, 4) or torch tensor
    Returns:
        quaternion (4,), translation (3,)
    """
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose).float()
    
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    
    quaternion = transforms3d.matrix_to_quaternion(rotation)
    
    return quaternion, translation

def transform_pose_to_original_frame(pose, transform_params):
    """
    Transform pose from DFNet frame back to original dataset frame
    
    For DFNet, we ONLY applied recentering, so we ONLY need to undo that.
    """
    pose = pose.clone()
    pose_matrix = pose.reshape(3, 4)
    
    # Inverse of recentering: subtract the move vector
    if transform_params['move_all_cam_vec'] != [0., 0., 0.]:
        move_vec = torch.tensor(transform_params['move_all_cam_vec'])
        pose_matrix[:3, 3] -= move_vec
    
    return pose_matrix

def transform_pose_to_dfnet_frame(pose, transform_params):
    """
    Transform pose from original dataset frame to DFNet frame
    
    For DFNet, we ONLY apply recentering (NO scaling)
    """
    pose = pose.clone()
    pose_matrix = pose.reshape(3, 4)
    
    # Apply recentering
    if transform_params['move_all_cam_vec'] != [0., 0., 0.]:
        move_vec = torch.tensor(transform_params['move_all_cam_vec'])
        pose_matrix[:3, 3] += move_vec
    
    return pose_matrix

def transform_pose_to_nerf_frame(pose, transform_params):
    """
    Transform pose from original dataset frame to NeRF frame
    
    This applies the SAME transformation as during training:
    Original → [scale, move, scale2] → NeRF frame
    """
    pose = pose.clone()
    pose_matrix = pose.reshape(3, 4)
    
    # Forward transform: same order as training
    # Step 1: Apply pose_scale
    pose_matrix[:3, 3] *= transform_params['pose_scale']
    
    # Step 2: Move to origin
    if transform_params['move_all_cam_vec'] != [0., 0., 0.]:
        move_vec = torch.tensor(transform_params['move_all_cam_vec'])
        pose_matrix[:3, 3] += move_vec * transform_params['pose_scale']
    
    # Step 3: Apply pose_scale2
    if transform_params['pose_scale2'] != 1.0:
        pose_matrix[:3, 3] *= transform_params['pose_scale2']
    
    # return pose_matrix.reshape(-1)
    return pose_matrix

def inference_single_image(model, image, device, preprocess=True):
    """
    Perform inference on a single image
    Args:
        model: trained DFNet model
        image: input image tensor (3, H, W)
        device: cuda device
        preprocess: whether to normalize input with ImageNet stats
    Returns:
        predicted pose as (3, 4) tensor
    """
    model.eval()
    
    # Add batch dimension
    image = image.unsqueeze(0).to(device)
    
    # Preprocess if needed (ImageNet normalization)
    if preprocess:
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
        image = (image - mean[None, :, None, None]) / std[None, :, None, None]
    
    with torch.no_grad():
        _, predicted_pose = model(image, return_feature=False)
        
        # Apply SVD to ensure valid rotation matrix
        predicted_pose = predicted_pose.reshape(1, 3, 4)
        R = predicted_pose[:, :3, :3]
        u, s, v = torch.svd(R)
        R_corrected = torch.matmul(u, v.transpose(-2, -1))
        predicted_pose[:, :3, :3] = R_corrected
    
    return predicted_pose.squeeze(0).cpu()


def evaluate_dataset(model, dataset_path, model_checkpoint, 
                     df=2.0, use_dfnet_s=False, preprocess=True,
                     save_results=True, output_dir='./inference_results',
                     compare_in_dfnet_frame=True,
                     world_setup_path=None):
    """
    Evaluate trained model on a dataset
    
    Args:
        compare_in_dfnet_frame: If True, transform GT to DFNet frame (centered)
                               If False, transform predictions to original frame
    """
    
    # Create dataset
    data_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    use_transforms = world_setup_path is not None
    
    dataset = CustomScenes(
        data_path=dataset_path,
        train=False,
        transform=data_transform,
        target_transform=target_transform,
        df=df,
        testskip=1,
        ret_idx=False,
        ret_hist=False,
        all_images=True,
        world_setup_path=world_setup_path
    )

    # Get transform parameters - for DFNet, only move_vec is used
    transform_params = {
        'move_all_cam_vec': dataset.move_all_cam_vec,
        'pose_scale': 1.0,  # NOT used for DFNet
        'pose_scale2': 1.0  # NOT used for DFNet
    }

    print(f"\nDFNet Transform parameters:")
    print(f"  move_all_cam_vec: {transform_params['move_all_cam_vec']}")
    print(f"  pose_scale: NOT USED (DFNet works at original scale)")
    print(f"  pose_scale2: NOT USED (only for NeRF)")
    print(f"  compare_in_dfnet_frame: {compare_in_dfnet_frame}")
    print(f"  use_transforms: {use_transforms}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"\nEvaluating on {len(dataset)} images from {dataset_path}")
    print(f"Image resolution: {dataset.H}x{dataset.W}")
    print(f"Using model: {model_checkpoint}")
    print("-" * 80)
    
    # Storage for results
    results = []
    translation_errors = []
    rotation_errors = []
    
    # Evaluate each image
    for idx, (image, gt_pose) in enumerate(tqdm(dataloader, desc="Processing images")):
        
        # Get prediction (in DFNet centered frame)
        pred_pose = inference_single_image(model, image[0], device, preprocess)
        
        gt_pose_matrix = gt_pose.reshape(3, 4)
        pred_pose_matrix = pred_pose.reshape(3, 4)
        
        # Transform to the same frame for comparison
        if use_transforms:
            if compare_in_dfnet_frame:
                # Transform GT to DFNet frame (centered, no scaling)
                gt_pose_matrix = transform_pose_to_dfnet_frame(gt_pose_matrix, transform_params)
                # pred_pose is already in DFNet frame
            else:
                # Transform prediction to original frame
                pred_pose_matrix = transform_pose_to_original_frame(pred_pose_matrix, transform_params)
                # gt_pose is already in original frame
        
        # Convert to quaternion + translation
        gt_quat, gt_trans = pose_matrix_to_quaternion_translation(gt_pose_matrix)
        pred_quat, pred_trans = pose_matrix_to_quaternion_translation(pred_pose_matrix)
        
        # Calculate errors
        rot_error = quaternion_angular_error(gt_quat, pred_quat)
        trans_error = translation_error(gt_trans, pred_trans)
        
        translation_errors.append(trans_error)
        rotation_errors.append(rot_error)
        
        # Store detailed results
        result = {
            'image_idx': idx,
            'image_path': dataset.rgb_files[idx],
            'translation_error_m': trans_error,
            'rotation_error_deg': rot_error,
            'gt_translation': gt_trans.numpy().tolist(),
            'pred_translation': pred_trans.numpy().tolist(),
            'gt_quaternion': gt_quat.numpy().tolist(),
            'pred_quaternion': pred_quat.numpy().tolist()
        }
        results.append(result)
    
    # Calculate statistics
    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors)
    
    comparison_frame = 'DFNet (centered)' if compare_in_dfnet_frame else 'Original'
    
    stats = {
        'dataset_path': dataset_path,
        'num_images': len(dataset),
        'comparison_frame': comparison_frame,
        'translation_error': {
            'mean': float(np.mean(translation_errors)),
            'median': float(np.median(translation_errors)),
            'std': float(np.std(translation_errors)),
            'min': float(np.min(translation_errors)),
            'max': float(np.max(translation_errors))
        },
        'rotation_error': {
            'mean': float(np.mean(rotation_errors)),
            'median': float(np.median(rotation_errors)),
            'std': float(np.std(rotation_errors)),
            'min': float(np.min(rotation_errors)),
            'max': float(np.max(rotation_errors))
        }
    }
    
    # Print results
    print("\n" + "=" * 80)
    print(f"RESULTS FOR {dataset_path}")
    print(f"Comparison Frame: {comparison_frame}")
    print("=" * 80)
    print(f"\nTranslation Error (meters):")
    print(f"  Mean:   {stats['translation_error']['mean']:.4f} m")
    print(f"  Median: {stats['translation_error']['median']:.4f} m")
    print(f"  Std:    {stats['translation_error']['std']:.4f} m")
    print(f"  Min:    {stats['translation_error']['min']:.4f} m")
    print(f"  Max:    {stats['translation_error']['max']:.4f} m")
    
    print(f"\nRotation Error (degrees):")
    print(f"  Mean:   {stats['rotation_error']['mean']:.2f}°")
    print(f"  Median: {stats['rotation_error']['median']:.2f}°")
    print(f"  Std:    {stats['rotation_error']['std']:.2f}°")
    print(f"  Min:    {stats['rotation_error']['min']:.2f}°")
    print(f"  Max:    {stats['rotation_error']['max']:.2f}°")
    print("=" * 80)
    
    # Save results to file
    if save_results:
        scene_name = osp.basename(osp.normpath(dataset_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary_path = osp.join(output_dir, f'{scene_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
        
        # Save detailed results
        detailed_path = osp.join(output_dir, f'{scene_name}_detailed.json')
        with open(detailed_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {detailed_path}")
        
        # Save as CSV for easy analysis
        csv_path = osp.join(output_dir, f'{scene_name}_results.csv')
        with open(csv_path, 'w') as f:
            f.write("image_idx,image_path,translation_error_m,rotation_error_deg\n")
            for r in results:
                f.write(f"{r['image_idx']},{r['image_path']},{r['translation_error_m']:.6f},{r['rotation_error_deg']:.4f}\n")
        print(f"CSV results saved to: {csv_path}")
    
    return stats, results


def main():
    parser = argparse.ArgumentParser(description='DFNet Inference and Evaluation')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--use_dfnet_s', action='store_true',
                       help='Use DFNet_s variant instead of DFNet')
    parser.add_argument('--preprocess', action='store_true', default=True,
                       help='Use ImageNet normalization (default: True)')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--df', type=float, default=2.0,
                       help='Downscale factor for images (default: 2.0)')
    
    # Comparison arguments
    parser.add_argument('--compare_in_original_frame', action='store_true',
                       help='Compare in original frame instead of NeRF frame')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--world_setup_path', type=str, default=None, 
                       help='Path to world_setup.json')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    if args.use_dfnet_s:
        model = DFNet_s()
    else:
        model = DFNet()
    
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Run evaluation
    evaluate_dataset(
        model=model,
        dataset_path=args.dataset_path,
        model_checkpoint=args.model_path,
        df=args.df,
        use_dfnet_s=args.use_dfnet_s,
        preprocess=args.preprocess,
        save_results=not args.no_save,
        output_dir=args.output_dir,
        compare_in_dfnet_frame=not args.compare_in_original_frame,  # Default: compare in NeRF frame
        world_setup_path=args.world_setup_path
    )


if __name__ == '__main__':
    main()