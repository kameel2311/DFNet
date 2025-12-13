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
                     save_results=True, output_dir='./inference_results'):
    """
    Evaluate trained model on a dataset
    Args:
        model: trained DFNet model
        dataset_path: path to dataset root
        scene_name: name of the scene
        model_checkpoint: path to model checkpoint
        df: downscale factor
        use_dfnet_s: whether using DFNet_s variant
        preprocess: whether to use ImageNet normalization
        save_results: whether to save detailed results to file
        output_dir: directory to save results
    """
    
    # Create dataset
    data_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
    
    dataset = CustomScenes(
        data_path=dataset_path,
        train=False,  # Use test split
        transform=data_transform,
        target_transform=target_transform,
        df=df,
        testskip=1,  # Don't skip any images
        ret_idx=False,
        ret_hist=False,
        all_images=True
    )
    
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
        
        # Get prediction
        pred_pose = inference_single_image(model, image[0], device, preprocess)
        
        # Convert to quaternion + translation
        gt_pose_matrix = gt_pose.reshape(3, 4)
        pred_pose_matrix = pred_pose.reshape(3, 4)
        
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
    
    stats = {
        'dataset_path': dataset_path,
        'num_images': len(dataset),
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
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    
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
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()