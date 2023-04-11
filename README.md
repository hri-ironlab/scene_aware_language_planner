# Generating Executable Action Plans with Environmentally-Aware Language Models

This is the official code and demo for the [Generating Executable Action Plans with Environmentally-Aware Language Models](https://arxiv.org/abs/2210.04964) paper. See [scene_aware_language_planner_demo.ipynb](https://github.com/hri-ironlab/scene_aware_language_planner/blob/main/src/scene_aware_language_planner_demo.ipynb) for instructions on running the demo.

Large Language Models (LLMs) trained using massive text datasets have recently shown promise in generating action plans for robotic agents from high level text queries. However, these models typically do not consider the robot’s environment, resulting in generated plans that may not actually be executable, due to ambiguities in the planned actions or environmental constraints. In this paper, we propose an approach to generate environmentally-aware action plans that agents are better able to execute. Our approach involves integrating environmental objects and object relations as additional inputs into LLM action plan generation to provide the system with an awareness of its surroundings, resulting in plans where each generated action is mapped to objects present in the scene. We also design a novel scoring function that, along with generating the action steps and associating them with objects, helps the system disambiguate among object instances and take into account their states. We evaluated our approach using the VirtualHome simulator and the ActivityPrograms knowledge base and found that action plans generated from our system had a 310% improvement in executability and a 147% improvement in correctness over prior work.
