import json
import textwrap
import os
import sys
physgaussian_path = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PhysGaussian", "mpm_solver_warp")
sys.path.append(physgaussian_path)
from mpm_solver_warp import (
    get_material_name, 
    MATERIAL_ID_TO_NAME,
    get_material_id,
)



INSTRUCTION_CONFIGS = {
    "tree": {
        "class_name_for_example": "ficus tree",
        "special_notes": "",
        "example_material_dict": {
            "pot": {"density": 400, "E": 2e8, "nu": 0.4, "material_id": get_material_id("stationary")},
            "trunk": {"density": 400, "E": 2e6, "nu": 0.4, "material_id": get_material_id("jelly")},
            "leaves": {"density": 200, "E": 2e4, "nu": 0.4, "material_id": get_material_id("jelly")}
        },
        "example_explanation": textwrap.dedent("""
            For this, we assume that the pot is stationary, while the trunk and leaves are made of "jelly", which will make
            them sway in the wind. The stiffness (Young's Modulus) of the trunk is much higher than that of the leaves.
        """),
        "example_all_queries": [["leaves", "trunk", "pot"], ["green", "orange", "reddish-brown"]],
        "tips": [
            "In a scene, typically there's a stationary part that will serve to fix the object to the ground. Usually, it's the pot, or some base of the tree. You must set the material_id of the stationary part to 6. If there's no stationary part, then never mind.",
            "For numerical stability, `E` should be between 1e4 and 1e6.",
            "The higher the `E` is, the stiffer the object is. E.g., so tree would sway less in the wind.",
        ],
        "example_constraints": textwrap.dedent("""
            assert material_dict["leaves"]["density"] < material_dict["trunk"]["density"] < material_dict["pot"]["density"], "The density of the leaves should be less than the trunk and the pot"
            assert material_dict["leaves"]["E"] < material_dict["trunk"]["E"] < material_dict["pot"]["E"], "The stiffness of the leaves should be less than the trunk and the pot"
        """),
    },
    "flowers": {
        "class_name_for_example": "flowers in a vase",
        "special_notes": "",
        "example_material_dict": {
            "vase": {"density": 500, "E": 1e6, "nu": 0.3, "material_id": get_material_id("stationary")},
            "flowers": {"density": 100, "E": 1e4, "nu": 0.4, "material_id": get_material_id("jelly")}
        },
        "example_explanation": textwrap.dedent("""
            Here, the vase is designated as stationary (material_id=6), indicating it should not move or sway.
            The flowers are set to a more pliable or flexible material (like "jelly" = 0), so that they can sway
            if there's wind or slight motion. The stiffness (Young's Modulus) of the vase is much higher than that
            of the flowers, making the vase rigid and the flowers more flexible.
        """),
        "example_all_queries": [["vase", "flowers"], ["ceramic base", "petals"], ["blue vase", "pink flower"]],
        "example_constraints": textwrap.dedent("""
            assert material_dict["vase"]["density"] > material_dict["flowers"]["density"], "The density of the vase should be greater than the flowers"
            assert material_dict["vase"]["E"] > material_dict["flowers"]["E"], "The stiffness of the vase should be greater than the flowers"
        """),
        "tips": [
            "In a typical flower arrangement, the vase (or base) is stationary, so give that part material_id=6 if present.",
            "For numerical stability, `E` should roughly be between 1e4 and 1e6.",
            "The higher the `E`, the stiffer the part. So the vase should have a higher E range than the flowers.",
        ]
    },
    "shrub": {
        "class_name_for_example": "typical three-part shrub",
        "special_notes": textwrap.dedent("""
            **Dataset note:** Shrubs in our dataset stand by themselves—there is **no planter or base**.
            You should therefore return only the shrub's structural parts and none of them are stationary.
        """),
        "example_material_dict": {
            "stems":    { "density": 300, "E": 1e5, "nu": 0.35, "material_id": get_material_id("jelly") },
            "twigs":    { "density": 250, "E": 6e4, "nu": 0.38, "material_id": get_material_id("jelly") },
            "foliage":  { "density": 150, "E": 2e4, "nu": 0.40, "material_id": get_material_id("jelly") }
        },
        "example_explanation": textwrap.dedent("""
            Return *ranges* instead of single values and accompany them with reasoning, Pythonic
            constraints, and alternative query lists.
        """),
        "example_all_queries": [
            ["stems", "twigs", "foliage"],
            ["woody stems", "thin branches", "leaves"],
            ["brown sticks", "small branches", "green leaves"]
        ],
        "tips": [
            "Provide exactly the parts visible (usually stems/twigs + foliage).",
            "1e4 <= E <= 1e6.",
            "Stems should be stiffest > twigs > foliage.",
            "No part uses material_id 6 because nothing is fixed to the ground.",
        ]
    },
    "grass": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            **Dataset note:** Grass patches are usually isolated; occasionally a visible soil patch is
            underneath. Include a "soil" part only if it is visible.
        """),
        "example_material_dict": {
            "blades": { "density": 80, "E": 1e4, "nu": 0.45, "material_id": get_material_id("jelly") }
        },
        "example_explanation": textwrap.dedent("""
            Example A (typical isolated grass—no stationary part):
            ```json
            {
                "blades": { "density": 80, "E": 1e4, "nu": 0.45, "material_id": get_material_id("jelly") }
            }
            ```

            Example B (grass with visible soil):
            ```json
            {
                "soil":   { "density": 1200, "E": 5e5, "nu": 0.30, "material_id": get_material_id("stationary") },
                "blades": { "density":  80,  "E": 1e4, "nu": 0.45, "material_id": get_material_id("jelly") }
            }
            ```
            Return *ranges*, reasoning, constraints, and alternative query lists.
        """),
        "example_all_queries": [
          ["blades"],
          ["grass"],
          ["green stalks"]
        ],
        "tips": [
            "Segment only the visible parts (sometimes just \"blades\").",
            "If *no* soil visible:\nall_queries: [[\"blades\"],[\"grass\"],[\"green stalks\"]]",
            "If soil *is* visible:\nall_queries: [[\"soil\", \"blades\"],[\"dirt\", \"grass\"],[\"brown base\", \"green grass\"]]",
            "1e4 <= E <= 1e6.",
            "If soil present -> give it material_id 6 and ensure E_soil > E_blades.",
            "If soil absent -> no stationary part; material_id 6 should not appear.",
        ]
    },
    "rubber_ducks_and_toys": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For rubber ducks and toys, we want to treat the entire object as a single part. Do not attempt to
            segment it into multiple parts. The object should be treated as a single, bouncy rubber-like object.
        """),
        "example_material_dict": {
            "toy": {"density": [80, 150], "E": [3e4, 5e4], "nu": [0.4, 0.45], "material_id": get_material_id("jelly")}
        },
        "example_explanation": "",
        "example_all_queries": [["toy"], ["rubber toy"], ["yellow duck"], ["plastic toy"]],
        "tips": [
            "Always use material_id=0 (jelly) for bouncy rubber-like behavior",
            "Keep E relatively low (around 1e3) for good bounce",
            "Density should be in the range of typical rubber/plastic toys",
            "Poisson's ratio should be around 0.35 for rubber-like behavior",
            "Make sure all queries in all_queries list are single-part queries"
        ]
    },
    "sport_balls": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For sport balls, we want to treat the entire ball as a single part. Do not attempt to
            segment it into multiple parts (like surface patterns or seams). The ball should be treated as a single,
            bouncy object.
        """),
        "example_material_dict": {
            "ball": {"density": [80, 150], "E": [3e4, 5e4], "nu": [0.4, 0.45], "material_id": get_material_id("jelly")}
        },
        "example_explanation": "",
        "example_all_queries": [["ball"], ["sport ball"], ["basketball"], ["round ball"]],
        "tips": [
            "Always use material_id=0 (jelly) for bouncy behavior",
            "Keep E relatively low (around 1e3) for good bounce",
            "Density should be in the range of typical sport balls",
            "Poisson's ratio should be around 0.35 for rubber-like behavior",
            "Make sure all queries in all_queries list are single-part queries"
        ]
    },
    "soda_cans": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For soda cans, we want to treat the entire can as a single part. Do not attempt to
            segment it into multiple parts (like the top, body, or label). The can should be treated as a single,
            rigid metal object.
        """),
        "example_material_dict": {
            "can": {"density": [2600, 2800], "E": [5e10, 8e10], "nu": [0.25, 0.35], "material_id": get_material_id("metal")}
        },
        "example_explanation": "",
        "example_all_queries": [["can"], ["soda can"], ["aluminum can"], ["metal can"]],
        "tips": [
            "Always use material_id=1 (metal) for rigid metal behavior",
            "Keep E relatively high (around 1e8) for metal stiffness",
            "Density should be in the range of typical aluminum (around 2700 kg/m³)",
            "Poisson's ratio should be around 0.3 for metal behavior",
            "Make sure all queries in all_queries list are single-part queries"
        ]
    },
    "metal_crates": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For metal crates, we want to treat the entire crate as a single part. Do not attempt to
            segment it into multiple parts (like the sides, top, or bottom). The crate should be treated as a single,
            rigid metal object.
        """),
        "example_material_dict": {
            "crate": {"density": [2500, 2900], "E": [8e7, 1.2e8], "nu": [0.25, 0.35], "material_id": get_material_id("metal")}
        },
        "example_explanation": "",
        "example_all_queries": [["crate"], ["metal crate"], ["metal box"], ["steel crate"]],
        "tips": [
            "Always use material_id=1 (metal) for rigid metal behavior",
            "Keep E relatively high (around 1e8) for metal stiffness",
            "Density should be in the range of typical metal (around 2700 kg/m³)",
            "Poisson's ratio should be around 0.3 for metal behavior",
            "Make sure all queries in all_queries list are single-part queries"
        ]
    },
    "sand": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For sand objects, we want to treat the entire object as a single part. Do not attempt to
            segment it into multiple parts. The sand should be treated as a single, granular material.
        """),
        "example_material_dict": {
            "sand": {"density": [1800, 2200], "E": [4e7, 6e7], "nu": [0.25, 0.35], "material_id": get_material_id("sand")}
        },
        "example_explanation": "",
        "example_all_queries": [["sand"], ["sand pile"], ["sand mound"], ["granular material"]],
        "tips": [
            "Always use material_id=2 (sand) for granular behavior",
            "Keep E relatively high (around 5e7) for sand stiffness",
            "Density should be in the range of typical sand (around 2000 kg/m³)",
            "Poisson's ratio should be around 0.3 for sand behavior",
            "Make sure all queries in all_queries list are single-part queries"
        ]
    },
    "jello_block": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For jello blocks, we want to treat the entire object as a single part. Do not attempt to
            segment it into multiple parts. The jello block should be treated as a single, soft, bouncy object.
        """),
        "example_material_dict": {
            "jello": {"density": [40, 60], "E": [800, 1200], "nu": [0.25, 0.35], "material_id": get_material_id("jelly")}
        },
        "example_explanation": "",
        "example_all_queries": [["jello"], ["jello block"], ["gelatin"], ["bouncy block"]],
        "tips": [
            "Always use material_id=0 (jelly) for soft, bouncy behavior",
            "Keep E relatively low (around 1000) for good bounce and jiggle",
            "Density should be in the range of typical jello (around 50 kg/m³)",
            "Poisson's ratio should be around 0.3 for jello-like behavior",
            "Make sure all queries in all_queries list are single-part queries"
        ]
    },
    "snow_and_mud": {
        "class_name_for_example": "",
        "special_notes": textwrap.dedent("""
            IMPORTANT: For combined snow & mud objects, we treat the entire mixture as a single deformable part.  Do **not**
            attempt to split it into separate snow and mud regions—the simulation will use one MPM material.
        """),
        "example_material_dict": {
            "snow_and_mud": {"density": [2000, 3000], "E": [8e4, 1.2e5], "nu": [0.15, 0.25], "material_id": get_material_id("snow")}
        },
        "example_explanation": "",
        "example_all_queries": [["snow and mud"], ["slush"], ["muddy snow"], ["wet snow"]],
        "tips": [
            "Always set material_id = 5 (snow) so the simulator uses the appropriate elasto-plastic snow model.",
            "Keep E around 1e5 (the config value) to match the intended softness.",
            "Density is markedly higher than fluffy snow because of the mud/water content—use roughly 2–3 g/cm³ (2000–3000 kg/m³).",
            "Make sure every list in `all_queries` contains **one** phrase because this is a single-part object."
        ]
    },
}


SYSTEM_INSTRUCTION_TEMPLATE = textwrap.dedent("""\
    We are trying to label a 3D object with physical properties. The physical properties are:
    - Density
    - Young's Modulus
    - Poisson's Ratio
    - Material model

    where the material model is one of the following:
{material_list_str}

    We have an automatic semantic segmentation model that can segment the object into different parts. We'll assume
    that each part has the same material model.

    Your job is to come up with the part query to pass to the semantic segmentation model, and the associated
    material properties for each part.
    {special_notes}
    For example, for a {class_name_for_example}, the return is

    ```json
    {example_material_dict_str}
    ```
    {example_explanation}
    Note that there are many different valid values for the material properties including E, nu, and density
    that would influence how the object behaves. Thus, instead of actual values, you should return
    a range of values like "E": [2e4, 2e6]. Also, provide reasoning and constraints on the values when appropriate.

    So the output should be a json with the following format:

    ```json
    {{
        "material_dict": {{ ... similar to example_dict with ranges ... }},
        "reasoning": "...",
        "constraints": "...",
        "all_queries": "..."
    }}
    ```

    Remember to write constraints in the form of python code. For example,
    ```python
    {example_constraints_str}
    ```

    Note that you've been asked to generate a material range so `material_dict["leaves"]["density"]` is a range of values. But for the purpose
    of the constraints writing, you can assume that the material_dict["leaves"]["density"] is a single value, and generate the python code similar
    to the example above. This is important because we will first sample a value from the range, then invoke your constraints code. So instead of writing something like
    ```python
    assert material_dict["leaves"]["density"][0] ...
    ```
    you must write something like
    ```python
    assert material_dict["leaves"]["density"] ...
    ```
    Note that the correct code doesn't have the bracket because `material_dict["leaves"]["density"]` will be already reduced to a single value by our sampler.
""")

PART_QUERY_INSTRUCTION_TEMPLATE = textwrap.dedent("""\
    You will be provided with images of the object from different views or a single view. Please try your best to come up with appropriate
    part queries as well. For example, if the object doesn't have visible trunk or pot, then you should
    NOT include them in the material_dict. Only segment parts that are visible in the image.

    Also, because our CLIP segmentation model is not perfect, you should come up with alternative queries as well including the original queries in the all_queries list.
    For example,
    ```json
    {example_all_queries_str}
    ```
    In total, you need to provide {num_alternative_queries} alternative queries. 

    Tips:
{tips_str}
    - Make sure that each element in the `all_queries` list is in the exact same order as the material_dict keys.
""")


def generate_instruction(class_name: str, num_alternative_queries: int) -> str:
    """
    Generates a complete instruction prompt for a given object class using a template
    and a configuration dictionary.
    """
    config = INSTRUCTION_CONFIGS[class_name]

    material_list_str = "\n".join([f"        {i}: \"{name}\"," for i, name in MATERIAL_ID_TO_NAME.items()])
    example_material_dict_str = json.dumps(config["example_material_dict"], indent=4)
    tips_str = "\n".join([f"    - {tip}" for tip in config["tips"]])
    example_all_queries_str = json.dumps(config["example_all_queries"], indent=4)

    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
        material_list_str=material_list_str,
        special_notes=config["special_notes"],
        class_name_for_example=config["class_name_for_example"] or class_name,
        example_material_dict_str=example_material_dict_str,
        example_explanation=config["example_explanation"],
        example_constraints_str=config.get("example_constraints", "..."),
    )

    part_query_instruction = PART_QUERY_INSTRUCTION_TEMPLATE.format(
        num_alternative_queries=num_alternative_queries,
        example_all_queries_str=example_all_queries_str,
        tips_str=tips_str
    )

    return system_instruction + part_query_instruction


INSTRUCTION_FUNCTIONS = {
    class_name: (lambda c: lambda n: generate_instruction(c, n))(class_name)
    for class_name in INSTRUCTION_CONFIGS
}