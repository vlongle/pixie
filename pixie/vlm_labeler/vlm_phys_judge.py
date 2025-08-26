import os
import argparse
import logging
from typing import List
from vlmx.agent import Agent, AgentConfig
from vlmx.utils import save_json, join_path, get_frames_from_video
from PIL import Image  # noqa: F401  # (import required for type checking in prompt utils)
import json
from pixie.utils import str2bool, set_logger


PHYSICS_JUDGE_SYSTEM_INSTRUCTION = """
You are a physics-realism judge for animation videos.

You will be shown several candidate animations of the SAME 3D object responding to the SAME textual prompt that describes its intended physical motion.

Your tasks:
1. Carefully watch each candidate animation.
2. Describe what's going on in the animation.
3. Evaluate how physically realistic the motion looks (0-5 scale).
4. Identify concrete pros / cons affecting the score (e.g. energy conservation errors, temporal jitter, incorrect response to gravity, static etc.).
5. Suggest specific improvements.
6. Pick the overall best candidate.

Please output ONLY valid JSON with the following schema:
{
  "candidate_evaluations": {
    "candidate_0": {"description": str, "score": float, "pros": str, "cons": str, "suggested_improvements": str},
    "candidate_1": { ... },
    "candidate_2": { ... }
  },
  "best_candidate": "candidate_i",   // the key of the best candidate
  "general_comments": str              // any overall remarks (optional)
}

Also, note that the first candidate is the "ground-truth", the gold standard, which should always be given a score of 5.
The other candidates should be judged based on how close they are to the ground-truth.

NOTE: ignore missing videos. Still return score for `candidate_{idx}` that are present.
"""

COTRACKER_INSTRUCTION = """
NOTE: to make your job easier, we have also annotated the ground-truth video with the Co-Tracker. Cotracker is a motion tracker algorithm to highlight the moving parts in the videos. 
Pay close attention to the motion traces annotated in the videos to gain information on how the object is moving.
Note that for objects that barely move, there will still be dots in the Co-Tracker video, but the motion
(lines) will be very short or non-existent, indicating that the points are not moving.
"""


class PhysicsJudgeAgent(Agent):
    OUT_RESULT_PATH = "vlm_phys_judge_results.json"

    def __init__(self, cfg: AgentConfig, num_frames: int = 8,
                 use_cotracker=False):
        self.num_frames = num_frames
        self.use_cotracker = use_cotracker
        super().__init__(cfg)

    # --------------------------------------------------
    # Agent interface implementation
    # --------------------------------------------------
    def _make_system_instruction(self):
        sys_instruction = PHYSICS_JUDGE_SYSTEM_INSTRUCTION
        logging.info("USING COTRACKER", self.use_cotracker)
        if self.use_cotracker:
            sys_instruction += COTRACKER_INSTRUCTION
        return sys_instruction

    def _make_prompt_parts(self, prompt: str, candidate_video_paths: List[str], **kwargs):
        """Build the multimodal prompt parts.

        Args:
            prompt (str): The textual description of the intended motion.
            candidate_video_paths (List[str]): 
        Returns:
            List: A list combining text strings and PIL Images as accepted by the underlying VLM.
        """
        # Allow variable number of candidates (≥2). The first video must be the ground-truth.
        if len(candidate_video_paths) < 2:
            raise ValueError("At least two videos (ground-truth + one candidate) are required")

        prompt_parts: List = []
        prompt_parts.append(
            "You are shown these candidate videos for the following prompt describing desired motion:\n'{}'\nEvaluate each candidate for physical realism.".format(prompt)
        )

        for idx, video_path in enumerate(candidate_video_paths):
            prompt_parts.append(f"Candidate {idx}:")
            try:
                frames = get_frames_from_video(
                    video_path,
                    num_frames=self.num_frames,
                    video_encoding_strategy="individual",
                    to_crop_white=False,
                    flip_horizontal=False,
                    width=256,
                    height=None,
                )
                logging.info(f"Loaded {len(frames)} frames from {video_path}")
                prompt_parts.extend(frames)
            except Exception as e:
                logging.error(f"Failed to load frames from {video_path}: {e}")
                prompt_parts.append(f"(Could not load video {video_path})")

        prompt_parts.append(
            "Provide your JSON answer following the required schema."
        )
        return prompt_parts

    def parse_response(self, response):
        """Parse and save the JSON returned by the model."""
        response_text = response.text.strip()
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object detected in the response")
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            parsed = {"error": str(e), "raw_response": response_text}

        out_path = join_path(self.cfg.out_dir, self.OUT_RESULT_PATH)
        save_json(parsed, out_path)
        logging.info(f"Saved results to {out_path}")
        return parsed


# --------------------------------------------------
# CLI entry-point
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge physical realism of 3 candidate videos")
    parser.add_argument("--prompt", type=str, required=True, help="Textual prompt used to generate the motion")
    parser.add_argument(
        "--candidate_videos",
        type=str,
        nargs='+',
        required=True,
        help="Paths to the candidate videos (space-separated). First video must be ground-truth.",
    )
    parser.add_argument("--out_dir", type=str, default="vlm_phys_judge_results")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames extracted from each video for the prompt")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Overwrite existing results if present")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash", help="Model name")
    parser.add_argument("--api_key", type=str, default="", help="API key (if not provided, will use environment variable)")
    parser.add_argument("--use_cotracker", type=str2bool, default=False, help="Use Co-Tracker to highlight the moving parts in the videos")
    # parser.add_argument("--model_name", type=str, default="o3", help="Model name")
    args = parser.parse_args()
    set_logger()

    agent = PhysicsJudgeAgent(
        AgentConfig(
            # model_name="gemini-2.0-flash",  # quick & cheap
            # model_name="gemini-2.5-pro-preview-03-25",  # higher quality for judging
            # model_name="o3",  # quick & cheap
            model_name=args.model_name,
            out_dir=args.out_dir,
            api_key=args.api_key,
        ),
        num_frames=args.num_frames,
        use_cotracker=args.use_cotracker,
    )

    config = {
        "temperature": 1.0 if args.model_name == "o3" else 0.5,
    }
    agent.generate_prediction(
        args.prompt,
        args.candidate_videos,
        overwrite=args.overwrite,
        gen_config=config,
    ) 