from typing import Dict, Type, Callable, Any, Optional, List, Union
from pathlib import Path
from IPython.display import display, Image as IPyImage
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass
from vlmx.utils import file_to_string, get_frames_from_video


@dataclass
class ArtifactPaths:
    """Paths to different types of artifacts in an iteration"""
    workspace: Path
    images: Path
    videos: Path
    logs: Path
    code: Path
    stdout: Path
    stderr: Path

    @classmethod
    def from_iteration_dir(cls, iter_dir: Path) -> 'ArtifactPaths':
        workspace = iter_dir / "vlm_workspace"
        return cls(
            workspace=workspace,
            images=workspace / "images",
            videos=workspace / "videos",
            logs=workspace / "logs",
            code=iter_dir / "tool_use_agent.py", ## NOTE: remember to change this to the correct code file
            stdout=iter_dir / "stdout.txt",
            stderr=iter_dir / "stderr.txt"
        )


class ArtifactHandler:
    """Base class for handling artifacts (display and processing)"""
    
    @staticmethod
    def image_to_base64(img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def read_file(path: Path) -> Optional[str]:
        """Safely read a file's contents"""
        try:
            return file_to_string(path)
        except Exception as e:
            # print(f"Error reading file {path}: {e}")
            return None


class ArtifactDisplayHandler(ArtifactHandler):
    """Handles the display of different types of artifacts"""
    
    def __init__(self, iter_dir: Optional[Path] = None):
        self.handlers: Dict[str, Callable] = {
            '.png': self.display_image,
            '.jpg': self.display_image,
            '.jpeg': self.display_image,
            '.json': self.display_json,
            '.py': self.display_code,
            '.txt': self.display_text,
            '.mp4': self.display_video,
        }
        self.paths = ArtifactPaths.from_iteration_dir(iter_dir) if iter_dir else None

    def display_image(self, path: Path) -> None:
        """Display image files"""
        display(IPyImage(filename=str(path)))
    
    def display_json(self, path: Path) -> None:
        """Display JSON files with pretty formatting"""
        content = self.read_file(path)
        if content:
            print(json.dumps(json.loads(content), indent=2))
    
    def display_code(self, path: Path) -> None:
        """Display code files with syntax highlighting if available"""
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalFormatter
            
            content = self.read_file(path)
            if content:
                print(highlight(content, PythonLexer(), TerminalFormatter()))
        except ImportError:
            self.display_text(path)
    
    def display_text(self, path: Path) -> None:
        """Display text files"""
        content = self.read_file(path)
        if content:
            print(content)
    
    def display_video(self, path: Path) -> None:
        """Display video as frames"""
        frames = get_frames_from_video(str(path))
        for i, frame in enumerate(frames):
            display(IPyImage(frame))
    
    def register_handler(self, extension: str, handler: Callable) -> None:
        """Register a new file handler"""
        self.handlers[extension.lower()] = handler
    
    def display_artifact(self, path: Path) -> None:
        """Display a single artifact"""
        extension = path.suffix.lower()
        if extension in self.handlers:
            print(f"\n📁 {path.name}")
            self.handlers[extension](path)
        else:
            print(f"\n📁 {path.name} (No display handler available)")
    


    def display_artifacts(self, directory: Path) -> None:
        """Display all artifacts in a directory"""
        if not directory.exists():
            print(f"Directory {directory} does not exist!")
            return
            
        files = sorted(directory.glob("*"))
        if not files:
            print(f"No artifacts found in {directory}")
            return
            
        print(f"\n=== Artifacts in {directory.name} ===")
        for file_path in files:
            self.display_artifact(file_path)
            

    def display_all(self) -> None:
        """Display all artifacts from the iteration directory"""
        if not self.paths:
            raise ValueError("No iteration directory was provided")
            
        print("\n=== Code ===")
        self.display_artifact(self.paths.code)
        
        print("\n=== Standard Output ===")
        self.display_artifact(self.paths.stdout)
        
        print("\n=== Standard Error ===")
        self.display_artifact(self.paths.stderr)
        
        print("\n=== Images ===")
        self.display_artifacts(self.paths.images)
        
        print("\n=== Videos ===")
        self.display_artifacts(self.paths.videos)
        
        print("\n=== Logs ===")
        self.display_artifacts(self.paths.logs)


class ArtifactCollector(ArtifactHandler):
    """Collects and processes artifacts for LLM context"""
    
    def __init__(self, iter_dir: Path):
        self.paths = ArtifactPaths.from_iteration_dir(iter_dir)

    def collect_images(self) -> List[Dict[str, str]]:
        """Collect and process image artifacts"""
        if not self.paths.images.exists():
            return []
        
        images = []
        for img_path in self.paths.images.glob("*.png"):
            try:
                img = Image.open(img_path)
                images.append({
                    'name': img_path.name,
                    # 'data': self.image_to_base64(img)
                    'data': img
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        return images

    def collect_videos(self) -> List[Dict[str, Any]]:
        """Collect and process video artifacts"""
        if not self.paths.videos.exists():
            return []
        
        videos = []
        for video_path in self.paths.videos.glob("*.mp4"):
            try:
                frames = get_frames_from_video(str(video_path))
                videos.append({
                    'name': video_path.name,
                    # 'frames': [self.image_to_base64(frame) for frame in frames]
                    'frames': frames,
                })
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
        return videos

    def collect_logs(self) -> List[Dict[str, Any]]:
        """Collect and process log artifacts"""
        if not self.paths.logs.exists():
            return []
        
        logs = []
        for log_path in self.paths.logs.glob("*.json"):
            content = self.read_file(log_path)
            if content:
                logs.append({
                    'name': log_path.name,
                    'data': json.loads(content)
                })
        return logs

    def collect(self) -> Dict[str, Any]:
        """Collect all artifacts and return organized context"""
        return {
            'code': self.read_file(self.paths.code),
            'stdout': self.read_file(self.paths.stdout),
            'stderr': self.read_file(self.paths.stderr),
            'images': self.collect_images(),
            'videos': self.collect_videos(),
            'logs': self.collect_logs()
        }


def artifacts_to_prompt_parts(artifacts: Dict[str, Any]) -> List[Union[str, Image.Image]]:
    """Convert collected artifacts into a list of prompt parts for the VLM.
    
    Args:
        artifacts: Dictionary of artifacts from ArtifactCollector.collect()
    
    Returns:
        List of strings and images that can be directly fed to the VLM
    """
    prompt_parts = []
    
    # Add code if available
    if artifacts['code']:
        prompt_parts.append(f"\nPrevious code:\n```python\n{artifacts['code']}\n```")
    
    # Add stderr if there were errors
    if artifacts['stderr']:
        prompt_parts.append(f"\nPrevious error:\n```\n{artifacts['stderr']}\n```")
    
    # Add stdout if available
    if artifacts['stdout']:
        prompt_parts.append(f"\nPrevious output:\n```\n{artifacts['stdout']}\n```")
    
    # Add images
    if artifacts['images']:
        prompt_parts.append("\nPrevious attempt images:")
        for img_obj in artifacts['images']:
            prompt_parts.extend([f"\n{img_obj['name']}:", img_obj['data']])
    
    # Add videos (as frames)
    if artifacts['videos']:
        prompt_parts.append("\nPrevious attempt video frames:")
        for video_obj in artifacts['videos']:
            prompt_parts.extend([f"\n{video_obj['name']} frames:"] + video_obj['frames'])
    
    # Add logs
    if artifacts['logs']:
        for log in artifacts['logs']:
            prompt_parts.append(
                f"\n{log['name']}:\n```json\n{json.dumps(log['data'], indent=2)}\n```"
            )
    
    return prompt_parts
