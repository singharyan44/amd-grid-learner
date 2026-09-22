"""LLM agent stub for M1 — rule fallback now, vLLM ROCm in M2.

M2 will replace `llm_policy` with a batched vLLM server call:
  fog_obs -> prompt JSON -> vLLM (ROCm wheel, MI300X) -> action
"""
from agents.agents import heuristic_action


def llm_policy(env, obs):
    # M1: fall back to heuristic so baselines run without GPU.
    return heuristic_action(env)


def obs_to_prompt(obs):
    return (
        f"pos={obs['pos']} steps_left={obs['steps_left']} "
        f"plates_done={obs['plates_done']} door_open={obs['door_open']} "
        f"fog={obs['fog']}\nAct: up/down/left/right/wait as JSON {{\"action\": ...}}"
    )
