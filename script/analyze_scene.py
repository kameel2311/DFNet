import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_scene(json_path):
    """Analyze scene bounds and camera poses"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    poses = []
    for frame in data['frames']:
        pose = np.array(frame['transform_matrix'])
        poses.append(pose[:3, 3])  # Extract translation
    
    poses = np.array(poses)
    
    # Calculate statistics
    min_bounds = poses.min(axis=0)
    max_bounds = poses.max(axis=0)
    center = poses.mean(axis=0)
    extent = max_bounds - min_bounds
    max_extent = extent.max()
    
    print("="*80)
    print("SCENE ANALYSIS")
    print("="*80)
    print(f"\nNumber of poses: {len(poses)}")
    print(f"\nBounds:")
    print(f"  X: [{min_bounds[0]:.3f}, {max_bounds[0]:.3f}] (extent: {extent[0]:.3f})")
    print(f"  Y: [{min_bounds[1]:.3f}, {max_bounds[1]:.3f}] (extent: {extent[1]:.3f})")
    print(f"  Z: [{min_bounds[2]:.3f}, {max_bounds[2]:.3f}] (extent: {extent[2]:.3f})")
    print(f"\nCenter: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"Max extent: {max_extent:.3f}")
    
    # Calculate recommended parameters
    # Goal: center at origin and scale to ~[-1, 1]
    move_vec = -center
    scale = 2.0 / max_extent  # Scale to fit in [-1, 1]
    
    print(f"\n" + "="*80)
    print("RECOMMENDED PARAMETERS FOR world_setup.json:")
    print("="*80)
    print(f"move_all_cam_vec: [{move_vec[0]:.6f}, {move_vec[1]:.6f}, {move_vec[2]:.6f}]")
    print(f"pose_scale: {scale:.6f}")
    print(f"pose_scale2: 1.0")
    
    # Estimate near/far based on scene depth
    near = max(0.01, extent.min() * 0.1)
    far = max_extent * 2
    print(f"near: {near:.3f}")
    print(f"far: {far:.3f}")
    
    # Visualize
    fig = plt.figure(figsize=(12, 5))
    
    # Original poses
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c='b', marker='o')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Camera Poses')
    
    # Transformed poses
    poses_transformed = (poses + move_vec) * scale
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(poses_transformed[:, 0], poses_transformed[:, 1], 
                poses_transformed[:, 2], c='r', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Transformed Poses (for NeRF)')
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('scene_analysis.png', dpi=150)
    print(f"\nVisualization saved to: scene_analysis.png")
    
    return {
        'move_all_cam_vec': move_vec.tolist(),
        'pose_scale': float(scale),
        'pose_scale2': 1.0,
        'near': float(near),
        'far': float(far)
    }

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_scene.py <path_to_transforms.json>")
        sys.exit(1)
    
    params = analyze_scene(sys.argv[1])
    
    # Save recommended world_setup.json
    output = {
        "near": params['near'],
        "far": params['far'],
        "pose_scale": params['pose_scale'],
        "pose_scale2": params['pose_scale2'],
        "move_all_cam_vec": params['move_all_cam_vec']
    }
    
    with open('recommended_world_setup.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"\nRecommended config saved to: recommended_world_setup.json")