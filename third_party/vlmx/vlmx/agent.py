from dataclasses import dataclass
from typing import Optional
from vlmx.utils import (
    create_dir,
    load_json,
    join_path,
    file_to_string,
)
from vlmx.prompt_utils import (
    setup_vlm_model,
    save_prompt_parts_as_html,
)
from omegaconf import OmegaConf
import os
import logging
from IPython.display import display, HTML

GEN_CONFIG = {
    "temperature": 0.5,
}


@dataclass
class AgentConfig:
    model_name: str
    out_dir: str
    api_key: Optional[str] = None


class Agent:
    def __init__(
        self, cfg: AgentConfig,
    ):
        self.cfg = cfg
        create_dir(cfg.out_dir)
        self.system_instruction = self.make_system_instruction()
        self.model = setup_vlm_model(
            model_name=cfg.model_name, system_instruction=self.system_instruction, api_key=cfg.api_key
        )
        print(f"Model: {self.model}")
        OmegaConf.save(self.cfg, join_path(cfg.out_dir, "config.json"))

    @ property
    def out_path(self):
        return join_path(self.cfg.out_dir, self.OUT_RESULT_PATH)

    @ property
    def error_path(self):
        return join_path(self.cfg.out_dir, "error.txt")

    def make_system_instruction(self):
        system_instruction = self._make_system_instruction()
        save_prompt_parts_as_html(
            system_instruction, join_path(
                self.cfg.out_dir, "system_instruction.html")
        )
        return system_instruction

    def load_system_instruction(self):
        return display(HTML(join_path(self.cfg.out_dir, "system_instruction.json")))

    def load_prompt_parts(self):
        return display(HTML(join_path(self.cfg.out_dir, "prompt.html")))

    def make_prompt_parts(self, *args, **kwargs):
        prompt_parts = self._make_prompt_parts(*args, **kwargs)
        save_prompt_parts_as_html(
            prompt_parts, join_path(self.cfg.out_dir, "prompt.html")
        )
        return prompt_parts

    def parse_response(self, response):
        raise NotImplementedError

    def _make_system_instruction(self):
        raise NotImplementedError

    def _make_prompt_parts(self, *args, **kwargs):
        raise NotImplementedError

    def generate_prediction(self, *args, gen_config=None, overwrite=False, **kwargs):
        out_path = join_path(self.cfg.out_dir, self.OUT_RESULT_PATH)
        if (
            os.path.exists(out_path)
            and not overwrite
        ):
            logging.info(
                f"{self.__class__.__name__}: Prediction already exists at {out_path}. Skipping generation."
            )
            return self.load_prediction()

        if gen_config is None:
            gen_config = GEN_CONFIG

        logging.info(f"{self.__class__.__name__}: Generating content.")
        prompt_parts = self.make_prompt_parts(*args, **kwargs)
        logging.info(f"Prompt: {prompt_parts}")

        response = self.model.generate_content(
            prompt_parts,
            generation_config=gen_config,
        )
        # logging.info(f"Usage: {response.usage_metadata}")

        self.parse_response(response, **kwargs)
        return response

    def load_prediction(self):
        if ".json" in self.OUT_RESULT_PATH:
            return load_json(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))
        return file_to_string(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))