from typing import Dict, Any, Optional
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
from vlmx.context_agent import ContextAwareAgent
import tempfile
from vlmx.utils import string_to_file, join_path


TOOL_INSTRUCTION = """
## Tool Use Instructions

When responding:

1. First explain your approach in plain text
2. Then provide your code in a single Python code block marked with ```python
3. Finally explain what the code does and what files were saved

Available tools:

1. save_image(name, image) -> saves matplotlib figure or numpy array as image
2. log_data(name, data) -> saves data as JSON file
Note: you can call the above tool-use functions directly as they'll be made available in the global scope. No need to import them.

Example response format:
I'll help you solve this problem by...
```python
[code]
```

The code will save the following files:
[files]

"""
PYTHON_DIFF_INSTRUCTION = """
## Diff instructions

When modifying existing code:
1. First explain your approach in plain text
2. Then provide your changes as a unified diff marked with ```diff
3. Finally explain what the changes do

The diff should be in standard unified diff format:
```diff
[diff]
```

As we will use this function to apply your diff:

```python
def apply_unified_diff(original_code: str, diff_text: str) -> str:
    "Applies a unified diff using the `patch` library."
    patch_set = patch.fromstring(diff_text.encode('utf-8'))  # Convert string to bytes
    patched_code = patch_set.apply(original_code.encode('utf-8'))
    return patched_code.decode('utf-8')  # Convert bytes back to string
```

Example response format:
I'll modify the code by...
```diff
[diff]
```

The changes will...



__Important Note for New Function Definitions__

When adding a new function definition immediately after another, ensure there's a blank line separating them. 
Otherwise, Python may raise indentation errors due to how the function boundaries are parsed.

"""



class ToolUseAgent(ContextAwareAgent):
    """
    An agent that can use tools and save/analyze artifacts during execution.
    Extends ContextAwareAgent to support file references in prompts.
    """
    
    def __init__(self, config, working_dir: Optional[str] = "./vlm_workspace"):
        super().__init__(config)
        self.working_dir = Path(join_path(self.cfg.out_dir, working_dir))
        self.working_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.img_dir = self.working_dir / "images"
        self.video_dir = self.working_dir / "videos" 
        self.log_dir = self.working_dir / "logs"
        self.code_dir = self.working_dir / "code"
        
        for dir in [self.img_dir, self.video_dir, self.log_dir, self.code_dir]:
            dir.mkdir(exist_ok=True)

    def save_image(self, name: str, image: Any) -> str:
        """Save image and return path that can be referenced with @"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.img_dir / f"{name}_{timestamp}.png"
        
        if isinstance(image, plt.Figure):
            # If it's a matplotlib figure, save it directly
            image.savefig(str(path))
            plt.close(image)  # Clean up the figure
        else:
            # Create a new figure and axis
            plt.figure(figsize=(8, 8))
            # Handle numpy arrays
            if hasattr(image, 'shape') and len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)
            plt.grid(True)
            # Show axis labels and ticks
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            plt.savefig(str(path))
            plt.close()  # Clean up the figure
        return str(path)

    def save_video(self, name: str, video_path: str) -> str:
        """Copy video to workspace and return path that can be referenced with @"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = self.video_dir / f"{name}_{timestamp}.mp4"
        import shutil
        shutil.copy2(video_path, new_path)
        return str(new_path)

    def log_data(self, name: str, data: Any) -> str:
        """Log data and return path that can be referenced with @"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"{name}_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return str(path)

    # def save_code(self, name: str, code: str) -> str:
    #     """Save code and return path that can be referenced with @"""
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     path = self.code_dir / f"{name}_{timestamp}.py"
    #     string_to_file(code, str(path))
    #     return str(path)

    def execute_code(self, code: str, globals_dict: Optional[Dict] = None) -> tuple[bool, Optional[str]]:
        """
        Execute code with provided globals and return success status and error message if any.
        The code can reference saved artifacts using @ syntax.
        """
        if globals_dict is None:
            globals_dict = {}
            
        # Add tool functions to globals
        globals_dict.update({
            'save_image': self.save_image,
            'save_video': self.save_video,
            'log_data': self.log_data,
            # 'save_code': self.save_code
        })
        
        try:
            # Save code to temporary file for potential reference
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_path = f.name
            
            # Execute the code
            exec(code, globals_dict)
            return True, None
        except Exception as e:
            import traceback
            return False, f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        finally:
            if 'code_path' in locals():
                import os
                os.unlink(code_path)

    def _make_system_instruction(self) -> str:
        return TOOL_INSTRUCTION


