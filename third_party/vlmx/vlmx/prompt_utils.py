import logging
import random
import astor
import ast
import re
import os
import google.generativeai as genai
from vlmx.utils import string_to_file
import textwrap
import markdown2
from io import BytesIO
import base64
from PIL import Image
import anthropic
from openai import OpenAI
 
# Add required imports at the top, after existing imports
import torch
import tempfile
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
import base64

def setup_gemini(model_name, system_instruction=None, api_key=None):
    logging.info(f"Setting up Google's model {model_name} with API key {api_key}")
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)

    # genai.list_models()
    return genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction,
    )




def setup_vlm_model(model_name, system_instruction=None, api_key=None):
    # return setup_gemini(model_name, system_instruction, api_key)
    # return setup_gpt(model_name, system_instruction, api_key)
    logging.info("VLM model name: {}".format(model_name))
    logging.info(f"Setting up VLM's model {model_name} with API key {api_key}")
    if "gemini" in model_name:
        return setup_gemini(model_name, system_instruction, api_key)
    elif "claude" in model_name:
        return setup_claude(model_name, system_instruction, api_key)
    elif "gpt" in model_name or 'o' in model_name:
        return setup_gpt(model_name, system_instruction, api_key)
    elif "qwen" in model_name.lower():
        return setup_qwen(model_name, system_instruction, api_key)
    else:
        raise ValueError("Model name must contain 'gpt' `claude` or `o` or 'gemini' or 'qwen'. Got: {}".format(model_name))

def setup_gemini(model_name, system_instruction=None, api_key=None):
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)

    # genai.list_models()
    return genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction,
    )


# Add this after setup_gpt

class QwenWrapper:
    def __init__(self, model_name, system_instruction, api_key=None):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28  # Reduced for memory efficiency
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def _save_pil_to_temp(self, pil_image):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        # Convert to RGB if the image is in a mode that can't be saved as JPEG
        # print(f"DEBUG: Image mode is: {pil_image.mode}")
        if pil_image.mode in ['RGBA', 'P', 'LA', 'PA']:
            # print(f"DEBUG: Converting from {pil_image.mode} to RGB")
            pil_image = pil_image.convert('RGB')
        # print(f"DEBUG: Final image mode before saving: {pil_image.mode}")
        pil_image.save(temp_path)
        return temp_path

    def generate_content(self, prompt_parts, generation_config={}):
        content = []
        temp_files = []
        for part in prompt_parts:
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
            elif isinstance(part, dict) and part.get("type") in ["image", "video"]:
                content.append(part)
            elif isinstance(part, Image.Image):
                temp_path = self._save_pil_to_temp(part)
                temp_files.append(temp_path)
                content.append({"type": "image", "image": temp_path})
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction}],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        max_new_tokens = 30720
        print(f"DEBUG: max_new_tokens: {max_new_tokens}")
        max_new_tokens = generation_config.get("max_new_tokens", max_new_tokens)
        temperature = generation_config.get("temperature", 0.5)
        do_sample = temperature > 0

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )

        generated_ids_trimmed = generated_ids[0, inputs.input_ids.shape[1] :]
        output_text = self.processor.decode(
            generated_ids_trimmed, skip_special_tokens=True
        )

        # Clean up temp files
        for f in temp_files:
            os.remove(f)

        class MockResponse:
            def __init__(self, text):
                self.text = text

        return MockResponse(output_text)


def setup_qwen(model_name, system_instruction=None, api_key=None):
    return QwenWrapper(model_name, system_instruction, api_key)


class ClaudeWrapper:
    def __init__(self, model_name, system_instruction, api_key):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _encode_image_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def _format_content(self, prompt_parts):
        """Format content for Claude API, handling both text and images"""
        if not isinstance(prompt_parts, list):
            return [{"type": "text", "text": prompt_parts}]
        
        formatted_content = []
        
        for part in prompt_parts:
            if isinstance(part, str):
                formatted_content.append({
                    "type": "text",
                    "text": part
                })
            elif isinstance(part, Image.Image):
                base64_image = self._encode_image_to_base64(part)
                formatted_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            elif isinstance(part, dict):
                formatted_content.append(part)
        
        return formatted_content
    
    def generate_content(self, prompt_parts, generation_config={}):
        temperature = generation_config.get("temperature", 0)
        max_tokens = generation_config.get("max_tokens", 3072)
        
        formatted_content = self._format_content(prompt_parts)
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.system_instruction,
            messages=[
                {
                    "role": "user",
                    "content": formatted_content
                }
            ]
        )
        print(">>> USAGE", message.usage)
        
        class MockResponse:
            def __init__(self, text):
                self.text = text
        
        return MockResponse(message.content[0].text)
    

def setup_claude(model_name, system_instruction=None, api_key=None):
    """
    Setup function for Claude models.
    
    Args:
        model_name (str): Name of the Claude model (e.g., "claude-3-opus-20240229")
        system_instruction (str, optional): System instruction for the model
        api_key (str, optional): Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY in environment
        
    Returns:
        ClaudeWrapper or None: Initialized wrapper if successful, None if no API key available
    """
    logging.info(f"Setting up Claude's model {model_name} with API key {api_key}")
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None
    
    return ClaudeWrapper(
        model_name=model_name,
        system_instruction=system_instruction,
        api_key=api_key
    )




class GPTWrapper:
    def __init__(self, model_name, system_instruction, api_key):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def _format_content(self, prompt_parts):
        """Format content for OpenAI API, handling both text and images"""
        if not isinstance(prompt_parts, list):
            return prompt_parts  # Return as is if it's just a string

        formatted_content = []
        
        for part in prompt_parts:
            if isinstance(part, str):
                formatted_content.append(part)
            elif isinstance(part, Image.Image):
                # Convert PIL Image to base64 and format for OpenAI
                base64_image = self._encode_image_to_base64(part)
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                })
            elif isinstance(part, dict) and part.get("type") == "image_url":
                # If it's already in the correct format, pass it through
                formatted_content.append(part)
        
        return formatted_content

    def generate_content(self, prompt_parts, generation_config={}):
        temperature = generation_config.get("temperature", 0.5)
        
        # Format the content for OpenAI
        formatted_content = self._format_content(prompt_parts)
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": self.system_instruction},
                {"role": "assistant", "content": self.system_instruction},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature)
            
        # Format response to match Gemini's format
        response_text = completion.choices[0].message.content
        
        # Create a MockResponse class to mimic the structure expected by parse_response
        class MockResponse:
            def __init__(self, text):
                self.text = text
        return MockResponse(response_text)

def setup_gpt(model_name, system_instruction=None, api_key=None):
    logging.info(f"Setting up OpenAI's model {model_name} with API key {api_key}")
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None
    
    return GPTWrapper(model_name=model_name, 
                      system_instruction=system_instruction, 
                      api_key=api_key)
    





def save_prompt_parts_as_html(prompt_parts, html_file_path):
    html_content = prompt_parts_to_html(prompt_parts)
    string_to_file(html_content, html_file_path)


def prompt_parts_to_html(prompt_parts, max_image_width=300, max_image_height=300):
    if isinstance(prompt_parts, (str, Image.Image)):
        # If a single string or image is passed, convert to a list
        prompt_parts = [prompt_parts]

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        <style>
            .image-row {{
                display: flex;
                flex-wrap: nowrap;
                margin-bottom: 20px;
            }}
            .image-row img {{
                margin-right: 1px;
                max-width: {max_width}px;
                max-height: {max_height}px;
                height: auto;
            }}
        </style>
    </head>
    <body>
    """.format(
        max_width=max_image_width, max_height=max_image_height
    )

    image_row_open = False

    for part in prompt_parts:
        if isinstance(part, str):
            if image_row_open:
                html_content += "</div>"
                image_row_open = False
            formatted_text = to_markdown(part)
            html_content += f"<p>{markdown2.markdown(formatted_text, extras=['fenced-code-blocks', 'code-friendly'])}</p>"
        elif isinstance(part, Image.Image):
            if not image_row_open:
                html_content += '<div class="image-row">'
                image_row_open = True
            buffered = BytesIO()
            part.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            html_content += f'<img src="data:image/png;base64,{img_str}">'

    if image_row_open:
        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    return html_content


def to_markdown(text):
    text = text.replace("•", "  *")
    return textwrap.indent(text, "> ", predicate=lambda _: True)




def remove_lines_containing(content: str, keyword: str) -> str:
    """Removes lines containing a specific keyword from the content string.

    Args:
        content (str): The content string to process.
        keyword (str): The keyword to search for.

    Returns:
        str: The content string with lines containing the keyword removed.
    """
    lines = content.split("\n")
    filtered_lines = [line for line in lines if keyword not in line]
    return "\n".join(filtered_lines)


def categorize_nodes(python_code: str):
    tree = ast.parse(python_code)

    imports = []
    other_top_level = []
    functions = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            other_top_level.append(node)

    return imports, other_top_level, functions


def create_new_module(imports, other_top_level, selected_functions):
    module = ast.Module(body=imports + other_top_level +
                        selected_functions, type_ignores=[])
    return astor.to_source(module)


def select_random_functions(functions, num_examples):
    num_examples = min(num_examples, len(functions))
    logging.info(
        f"Selecting {num_examples} examples")
    return random.sample(functions, num_examples)


def get_n_examples_from_python_code(python_code, num_examples):
    if isinstance(num_examples, int) and num_examples >= 0:
        imports, other_top_level, functions = categorize_nodes(python_code)
        selected_functions = select_random_functions(functions, num_examples)
        python_code = create_new_module(
            imports, other_top_level, selected_functions)
    return python_code