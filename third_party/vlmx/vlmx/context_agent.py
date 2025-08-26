from dataclasses import dataclass
from typing import List, Dict, Union
from vlmx.agent import Agent
from vlmx.utils import file_to_string
import logging
from PIL import Image
import re
import os

@dataclass
class FileContext:
    """Represents a file and its content"""
    file_path: str
    content: str

class ContextMixin:
    """A mixin that adds file context handling to any agent"""
    
    def __init__(self, *args, **kwargs):
        self.file_contexts: Dict[str, FileContext] = {}  # Track all seen file contexts
        super().__init__(*args, **kwargs)
    
    def add_file_context(self, text: str) -> str:
        """Process text for file contexts and return cleaned text"""
        # Split only on whitespace-bounded @file references
        file_references = re.finditer(r'\s@(\S+)', ' ' + text)
        
        # Collect all file paths and their positions
        replacements = []
        new_file_contexts = []
        for match in file_references:
            file_path = match.group(1)

            if file_path in self.file_contexts or file_path in new_file_contexts:
                continue

            if not os.path.exists(file_path):
                logging.error(f"File {file_path} does not exist")
                continue

            ## add the file context to the list
            content = file_to_string(file_path)
            self.file_contexts[file_path] = FileContext(file_path, content)
            new_file_contexts.append(FileContext(file_path, content))
            
            # Replace @file.py with [file.py]
            basename = os.path.basename(file_path)
            replacements.append((match.group(0), f' [{basename}]'))
        
        # Apply replacements in reverse order
        cleaned_text = text
        for old, new in sorted(replacements, key=lambda x: x[0], reverse=True):
            cleaned_text = cleaned_text.replace(old, new)
                
        cleaned_text = cleaned_text.strip()

        ## now prepend the file contexts to the cleaned text
        parts = []
        for ctx in new_file_contexts:
            parts.append(
                f"\nContent of `{ctx.file_path}`:\n```\n{ctx.content}\n```\n"
            )
        parts.append(cleaned_text)
        return "\n".join(parts)

class ContextAwareAgent(ContextMixin, Agent): 
    """A base agent class that's aware of file contexts"""

    def _make_prompt_parts(self, prompt_parts: Union[str, List[Union[str, Image.Image]]], 
                           *args, **kwargs) -> List[Union[str, Image.Image]]:

        if isinstance(prompt_parts, str):
            prompt_parts = [prompt_parts]

        cleaned_parts = []
        for part in prompt_parts:
            if isinstance(part, str):
                cleaned_parts.append(self.add_file_context(part))
            else:
                cleaned_parts.append(part)
        return cleaned_parts