import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import cv2
import sys
from collections import defaultdict
import subprocess
from typing import List, Dict, Any
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import os
import random
import logging
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import re


def extract_code_from_string(code_string, code_type="python"):
    """Extracts code or diff from a string."""
    pattern = f"```{code_type}\n([\s\S]*?)```"
    matches = re.findall(pattern, code_string, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return None



def seed_everything(seed: int, torch_deterministic=False) -> None:
    # import torch
    logging.info(f"Setting seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # if torch_deterministic:
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    #     torch.use_deterministic_algorithms(True)


def join_path(*args):
    return os.path.join(*args)


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def string_to_file(string: str, filename: str) -> None:
    with open(filename, 'w') as file:
        file.write(string)


def file_to_string(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read()


def create_task_config(cfg: DictConfig, task_name) -> DictConfig:
    task_config = deepcopy(cfg)
    task_config.out_dir = join_path(task_config.out_dir, task_name)
    return task_config


def load_config(config_path="../../conf", config_name="config"):
    """
    Load and merge Hydra configuration.

    :param config_path: Path to the config directory
    :param config_name: Name of the main config file (without .yaml extension)
    :return: Merged configuration object
    """
    # Initialize Hydra
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)

    # Compose the configuration
    cfg = compose(config_name=config_name)

    return cfg


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def config_to_command(cfg: DictConfig, script_path: str, conda_env: str = "articulate-anything-clean") -> List[str]:
    """
    Convert a configuration to a command-line command, flattening nested structures.

    Args:
    cfg (DictConfig): The configuration to convert.
    script_path (str): The path to the Python script to run.
    conda_env (str): The name of the Conda environment to use.

    Returns:
    List[str]: The command as a list of strings.
    """
    # Convert the configuration to a flat dictionary
    flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # Convert the flat dictionary to command-line arguments
    cmd_args = [f"{k}={v}" for k, v in flat_cfg.items() if v is not None]
    return make_cmd(script_path, conda_env, cmd_args)


def make_cmd(script_path: str, conda_env: str = "articulate-anything-clean",
             cmd_args=[]):
    # Construct the command
    command = [
        "conda", "run", "-n", conda_env,
        "python", script_path
    ] + cmd_args

    return command


def run_subprocess(command: List[str], env=None) -> None:
    """
    Run a command as a subprocess.

    Args:
    command (List[str]): The command to run as a list of strings.

    Raises:
    subprocess.CalledProcessError: If the command fails.
    """
    # convert all element in command to string
    command = [str(c) for c in command]
    if env is None:
        env = os.environ.copy()
    try:
        subprocess.run(command, check=True, env=env)

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e}")


class Steps:
    def __init__(self):
        self.steps = defaultdict(dict)
        self.order = []

    def add_step(self, name: str, result: Any):
        self.steps[name] = result
        self.order.append(name)

    def __getitem__(self, name):
        return self.steps[name]

    def __iter__(self):
        for name in self.order:
            yield name, self.steps[name]

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return str(self.steps)


class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)


def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count


def extract_frames(video_path, method="fixed", num_frames=5, interval=1, start_time=None, end_time=None):
    """
    Extract frames from a video either based on a fixed number of frames or at regular intervals.

    Parameters:
    - video_path (str): Path to the video file.
    - method (str): Method to extract frames ('fixed' or 'interval').
    - num_frames (int): Number of frames to extract (used if method is 'fixed'). If -1, returns all frames.
    - interval (int): Interval in seconds between frames (used if method is 'interval').
    - start_time (float): Start time in seconds. If None, starts from beginning.
    - end_time (float): End time in seconds. If None, goes until end.

    Returns:
    - frames (list): List of extracted frames.
    - frame_info (dict): Dictionary with video and frame extraction details.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert time to frame numbers
    start_frame = 0 if start_time is None else int(start_time * fps)
    end_frame = frame_count if end_time is None else int(end_time * fps)
    
    # Ensure valid frame range
    start_frame = max(0, min(start_frame, frame_count - 1))
    end_frame = max(start_frame + 1, min(end_frame, frame_count))
    
    frames = []

    if method == "fixed":
        if num_frames == -1:
            # Return all frames in the specified range
            sample_indices = list(range(start_frame, end_frame))
        else:
            # Sample a fixed number of frames from the range
            range_size = end_frame - start_frame
            sample_indices = [start_frame + int(range_size * i / num_frames)
                          for i in range(num_frames)]
    elif method == "interval":
        # Sample frames at regular intervals within the range
        current_time = start_time if start_time is not None else 0
        end_time = end_time if end_time is not None else duration
        sample_indices = [
            int(fps * t) for t in np.arange(current_time, end_time, interval)
        ]
        # Filter indices to be within valid range
        sample_indices = [idx for idx in sample_indices if start_frame <= idx < end_frame]
    else:
        raise ValueError("Invalid method. Use 'fixed' or 'interval'.")

    print("getting frames", sample_indices)
    for idx in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()

    frame_info = {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "extracted_frame_indices": sample_indices,
        "start_time": start_time,
        "end_time": end_time,
    }
    return frames, frame_info


def concatenate_frames_horizontally(frames):
    """
    Concatenates frames into a single image horizontally.

    Args:
        frames (list): List of PIL Images or numpy arrays to be concatenated.

    Returns:
        np.array: Concatenated image.
    """
    # Convert PIL Images to numpy arrays if necessary
    if isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]

    frames = np.array(frames)

    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(
            "The frames array must have shape (n, height, width, channels)."
        )

    concatenated_image = np.concatenate(frames, axis=1)
    return concatenated_image


def crop_white(image):
    """
    Crop white space from around a PIL image

    :param image: PIL Image object
    :return: Cropped PIL Image object
    """
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get the bounding box of the non-white area
    bg = Image.new(image.mode, image.size, (255, 255, 255))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()

    if bbox:
        return image.crop(bbox)
    return image  # return the original image if it's all white


def resize_frame(frame, width, height):
    if width is None and height is None:
        return frame

    original_width, original_height = frame.size

    if width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = original_width / original_height
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = original_height / original_width
        height = int(width * aspect_ratio)

    return frame.resize((width, height), Image.LANCZOS)


def get_frames_from_video(
    video_path,
    num_frames=5,
    video_encoding_strategy="individual",
    to_crop_white=True,
    flip_horizontal=False,
    width=None,
    height=None,
    start_time=None,
    end_time=None,
):
    """
    Extract frames from a video and process them according to specified parameters.
    
    Parameters:
    - video_path (str): Path to the video file
    - num_frames (int): Number of frames to extract. If -1, returns all frames
    - video_encoding_strategy (str): How to encode the video frames ('individual' or 'concatenate')
    - to_crop_white (bool): Whether to crop white space from frames
    - flip_horizontal (bool): Whether to flip frames horizontally
    - width (int): Target width for resizing
    - height (int): Target height for resizing
    - start_time (float): Start time in seconds. If None, starts from beginning
    - end_time (float): End time in seconds. If None, goes until end
    
    Returns:
    - list: List of processed frames as PIL Images
    """
    frames, _ = extract_frames(video_path, num_frames=num_frames, 
                             start_time=start_time, end_time=end_time)
    pil_frames = [Image.fromarray(frame) for frame in frames]

    if flip_horizontal:
        pil_frames = [frame.transpose(Image.FLIP_LEFT_RIGHT)
                      for frame in pil_frames]

    if to_crop_white:
        pil_frames = [frame.transpose(Image.FLIP_LEFT_RIGHT)
                      for frame in pil_frames]

    # Get the size of the first frame as reference
    if pil_frames:
        reference_size = pil_frames[0].size
        # Resize all frames to match the first frame's size
        pil_frames = [frame.resize(reference_size, Image.LANCZOS) for frame in pil_frames]

    if width is not None or height is not None:
        # Resize the frames if either width or height is specified
        pil_frames = [resize_frame(frame, width, height)
                      for frame in pil_frames]

    if video_encoding_strategy == "concatenate":
        return [Image.fromarray(concatenate_frames_horizontally(pil_frames))]
    elif video_encoding_strategy == "individual":
        return pil_frames
    else:
        raise ValueError(
            "Invalid video_encoding_strategy. Use 'concatenate' or 'individual'."
        )


def convert_mp4_to_gif(input_path, output_path, start_time=0, end_time=None, resize=None, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return output_path

    with HideOutput():
        # Load the video file
        clip = VideoFileClip(input_path)
        # .subclip(start_time, end_time)

        # Resize if needed
        if resize:
            clip = clip.resize(resize)

        # Attempt a simpler write_gif call
        clip.write_gif(output_path, fps=10)
    return output_path


def display_frames(
    frames,
    titles=None,
    cols=5,
    figsize=(20, 10),
    border_color=None,
    border_width=20,
    wspace=0.0,
    hspace=0.0,
    save_file=None,
):
    """
    Display a list of frames with optional titles and optional colored borders.

    Parameters:
    - frames (list): List of frames to display.
    - titles (list): Optional list of titles for each frame.
    - cols (int): Number of columns in the display grid.
    - figsize (tuple): Size of the figure.
    - border_color (str): Optional color for the border around each frame.
    - border_width (int): Width of the border around each frame.
    - wspace (float): Width space between subplots.
    - hspace (float): Height space between subplots.
    """
    num_frames = len(frames)
    # Calculate the number of rows needed
    rows = (num_frames + cols - 1) // cols

    if border_color:
        frames = [draw_frame(frame, border_color, border_width)
                  for frame in frames]

    plt.figure(figsize=figsize)
    plt.ioff()
    for i, frame in enumerate(frames):
        ax = plt.subplot(rows, cols, i + 1)
        ax.axis("off")
        plt.imshow(frame)
        if titles and i < len(titles):
            plt.title(titles[i])
    plt.axis("off")
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        print(f"Saved to {save_file}")
        plt.close()  # Close the figure after saving
    else:
        plt.show()


def show_video(video, overwrite=True, use_gif=False,
               num_frames=5, flip_horizontal=False):
    from IPython.display import Video, Image
    if use_gif:
        gif = convert_mp4_to_gif(video, video.replace(".mp4", ".gif"),
                                 overwrite=overwrite)
        display(Image(gif))
    else:
        frames = get_frames_from_video(video, to_crop_white=True,
                                       num_frames=num_frames,
                                       flip_horizontal=flip_horizontal)
        display_frames(frames, cols=5)


def display_overlay_frames(
    frames,
    x_delta: int = 0,  # Horizontal offset applied cumulatively to each subsequent frame
    y_delta: int = 0,  # Vertical offset applied cumulatively to each subsequent frame
    z_delta: int = 0,  # Placeholder for z-offset, currently unused in this 2D context
    figsize=(10, 10),
    save_file=None,
    initial_opacity=0.5,
):
    """
    Display frames overlaid on top of each other with increasing opacity for later frames.
    Frames can be offset using x_delta and y_delta.
    
    Parameters:
    - frames (list): List of frames to overlay (PIL Images or numpy arrays)
    - x_delta (int): Horizontal offset applied cumulatively to each subsequent frame (e.g., frame N is offset by N*x_delta).
    - y_delta (int): Vertical offset applied cumulatively to each subsequent frame (e.g., frame N is offset by N*y_delta).
    - z_delta (int): Placeholder for z-offset, currently unused.
    - figsize (tuple): Size of the figure
    - save_file (str): Optional path to save the output image
    """
    if not frames:
        return
        
    # Convert numpy arrays to PIL Images if needed, or copy PIL images
    temp_pil_frames = []
    if isinstance(frames[0], np.ndarray):
        temp_pil_frames = [Image.fromarray(frame) for frame in frames]
    else:
        # Ensure we are working with copies if input is already PIL Images
        temp_pil_frames = [frame.copy() for frame in frames]
    
    if not temp_pil_frames:
        return

    base_size = temp_pil_frames[0].size
    
    # Resize all frames to match the first frame's size and convert to RGBA for transparency handling
    pil_frames_rgba = [frame.resize(base_size, Image.LANCZOS).convert('RGBA') for frame in temp_pil_frames]
    
    # Start with the first frame (it's already RGBA)
    result = pil_frames_rgba[0].copy()
    
    # Overlay subsequent frames with increasing opacity and offsets
    # Iterate over the rest of the frames (from the second frame onwards)
    for i, current_frame_rgba in enumerate(pil_frames_rgba[1:], 1):
        # i will be 1 for pil_frames_rgba[1], 2 for pil_frames_rgba[2], etc.
        # This index `i` is used for cumulative offset and opacity calculation.
        opacity = min(initial_opacity + (i * 0.1), 1.0)
        
        # Create a transparent canvas for the current_frame_rgba, same size as base
        offset_canvas = Image.new('RGBA', base_size, (0, 0, 0, 0))
        
        # Calculate cumulative offset for current_frame_rgba (which is effectively original_frames[i])
        # The first frame (original_frames[0]) is the base and is not offset.
        # original_frames[1] (current_frame_rgba when i=1) is offset by (1*x_delta, 1*y_delta)
        current_x_offset = i * x_delta
        current_y_offset = i * y_delta
        
        # Paste the current_frame_rgba onto the offset_canvas using its own alpha as mask
        offset_canvas.paste(current_frame_rgba, (current_x_offset, current_y_offset), current_frame_rgba)
        
        # Blend the offset_canvas (containing the single, offsetted frame) with the accumulated result
        result = Image.blend(result, offset_canvas, opacity)
        ## do compositing
        # result = Image.composite(offset_canvas, result, offset_canvas)
    
    # Display the result
    plt.figure(figsize=figsize)
    plt.imshow(result)
    plt.axis('off')
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()