from pixie.voxel.segmentation import *
from pixie.utils import *
import matplotlib.pyplot as plt
from vlmx.utils import load_json


def visualize_part_segmentation(
    coords: torch.Tensor,
    part_labels: torch.Tensor,
    part_queries: List[str],
    part_scores: torch.Tensor = None,
    use_scores_for_alpha: bool = False,
    point_size: float = 5.0,
    figsize: Tuple[int, int] = (12, 10),
    view_angles: Tuple[float, float] = (30, 45),
    save_path: str = None,
):
    """
    Visualize part segmentation results with different colors for each part.
    
    Args:
        coords: Tensor of shape (num_points, 3) containing point coordinates
        part_labels: Tensor of shape (num_points) containing part indices
        part_queries: List of part names corresponding to the indices
        part_scores: Optional tensor of shape (num_points) with similarity scores
        use_scores_for_alpha: If True, use scores to determine point transparency
        point_size: Size of points in the scatter plot
        figsize: Figure size (width, height) in inches
        view_angles: Tuple of (elevation, azimuth) for the 3D view
        save_path: If provided, save the figure to this path
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Convert tensors to numpy arrays
    coords_np = coords.cpu().numpy()
    part_labels_np = part_labels.cpu().numpy()
    
    if use_scores_for_alpha and part_scores is not None:
        part_scores_np = part_scores.cpu().numpy()
    
    # Check if there are any -1 labels present
    has_unassigned = -1 in part_labels_np
    
    # Create a colormap with distinct colors for each part
    # Only add an extra color for the unassigned label if it exists
    num_parts = len(part_queries)
    num_colors_needed = num_parts + (1 if has_unassigned else 0)
    cmap = plt.colormaps['tab10']
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # First handle the unassigned (-1) labels if they exist
    if has_unassigned:
        mask_unassigned = (part_labels_np == -1)
        if np.any(mask_unassigned):
            unassigned_points = coords_np[mask_unassigned]
            # Use the last color in the colormap for unassigned points
            unassigned_color = cmap(num_colors_needed - 1)
            
            ax.scatter(
                unassigned_points[:, 0], 
                unassigned_points[:, 1], 
                unassigned_points[:, 2],
                color=unassigned_color,
                s=point_size,
                label="unassigned",
                alpha=0.5  # Lower alpha for unassigned points
            )
    
    # Plot each part with a different color and variable alpha if requested
    for i, part_name in enumerate(part_queries):
        mask = (part_labels_np == i)
        if not np.any(mask):
            continue
            
        part_points = coords_np[mask]
        base_color = cmap(i)  # RGBA tuple
        
        if use_scores_for_alpha and part_scores is not None:
            # Use the part scores to define alpha for each point.
            alphas = part_scores_np[mask]
            # Ensure alphas are in the valid range [0, 1]. If not, clip them.
            alphas = np.clip(alphas, 0, 1)
            # Create an array of colors: replicate the base RGB for each point and set the alpha channel.
            rgb = np.array(base_color[:3])
            colors = np.tile(rgb, (part_points.shape[0], 1))
            # Append the per-point alpha as the 4th channel.
            colors = np.concatenate([colors, alphas[:, None]], axis=1)
            ax.scatter(
                part_points[:, 0], 
                part_points[:, 1], 
                part_points[:, 2],
                c=colors,
                s=point_size,
                label=part_name,
            )
        else:
            # Use a fixed alpha if not using scores
            ax.scatter(
                part_points[:, 0], 
                part_points[:, 1], 
                part_points[:, 2],
                color=base_color,
                s=point_size,
                label=part_name,
                alpha=0.8
            )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('Part Segmentation')
    # ax.set_axis_off()
    
    # Set the viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Add a legend
    lgnd = ax.legend(fontsize=20)
    for handle in lgnd.legend_handles:
        handle.set_sizes([80])

    
    # Make axes equal for better visualization
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range) / 2.0
    
    x_mid = (x_limits[1] + x_limits[0]) / 2
    y_mid = (y_limits[1] + y_limits[0]) / 2
    z_mid = (z_limits[1] + z_limits[0]) / 2
    
    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])
    
    # Make sure all axes have equal scale
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, required=True,
                       default="ecb91f433f144a7798724890f0528b23")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--part_queries", type=str, required=True,
                        default="pot, trunk, leaves")
    parser.add_argument("--render_outputs_dir", type=str, required=True,
                        help="Directory containing render outputs (clip_features.npz, clip_features_pc.ply)")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--material_dict_path", type=str, default=None,
                        help="Path to JSON file mapping part queries to material properties")
    parser.add_argument("--overwrite", type=str2bool, default=False)
                        
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)



    if args.material_dict_path is not None:
        with open(args.material_dict_path, 'r') as f:
            material_props = json.load(f)
        if "material_dict" in material_props:
            material_props = material_props["material_dict"]

    out_fig_path = f"{args.output_dir}/clip.png"
    if not args.overwrite and os.path.exists(out_fig_path):
        exit(0)

    obj_id = args.obj_id
    part_queries = [q.strip() for q in args.part_queries.split(",")]

    result_dir = args.render_outputs_dir


    grid_feature_path = f"{result_dir}/clip_features.npz"
    occupancy_path = f"{result_dir}/clip_features_pc.ply"


    coords_filtered, part_labels, part_scores, metrics = clip_part_segmentation(
        grid_feature_path,
        part_queries,
        occupancy_path,
    )

    fig = visualize_part_segmentation(
        coords_filtered, 
        part_labels, 
        part_queries,
        part_scores=part_scores,
        use_scores_for_alpha=False,
        point_size=2.0,
        figsize=(14, 12),
        view_angles=(0, 0)
    )

    fig.savefig(out_fig_path,
                bbox_inches='tight',
                dpi=300,
                pad_inches=0,
    
    )
    logging.info(f"SAVED image to {args.output_dir}/clip.png")


    if args.material_dict_path is not None:
        save_segmented_point_cloud(coords_filtered, part_labels, args.output_dir, 
                                original_pc_path=occupancy_path,
                                part_queries=part_queries, 
                                material_props=material_props,
                                grid_feature_path=grid_feature_path)  # Pass grid_feature_path

    