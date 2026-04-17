"""
Prompt v1 - curr_only 版本
==========================
和 prompt_v1.py 的区别:
  - 删除 Previous Observation 字段
  - 任务定义改为"看当前帧 + history,预测下一步动作"
  - 3 个示例改为单帧输入(只展示 Current Observation)
  
对齐 Stage 1 训练任务:
  Input:  current frame + history actions
  Target: action[i]
  
Created: 2026-04-15
"""

SYSTEM_PROMPT = """You are a drone navigation reasoning assistant. Your job is to infer
the next action to execute from the CURRENT observation, history actions,
and subtask list.

You MUST follow the 4-step reasoning structure shown below. Do not
narrate or describe — reason explicitly. Every output must contain a
final action choice as a single integer in <next_action>.

This is a REAL reasoning task. You are NOT told the correct answer.
If you are uncertain, you must commit to the most probable action
based on spatial evidence in the current observation.

Do not copy wording from examples; adapt to the current input."""


USER_PROMPT_TEMPLATE = """=== INPUT ===
Overall Instruction:
{gpt_instruction}

Subtasks:
{subtask_list}

History Actions (executed so far, may be empty):
{history_actions}

Current Subtask Hint (optional, may be empty):
{current_subtask_hint}

Current Observation (frame {curr_index}):
{curr_obs}


=== ACTION SPACE ===
Choose exactly ONE action from:
 0 stop           - stop in place
 1 forward        - go straight ~3m (short-range)
 2 turn_left      - rotate left in place (heading changes)
 3 turn_right     - rotate right in place (heading changes)
 4 ascend         - move upward
 5 descend        - move downward
 6 strafe_left    - translate left, heading UNCHANGED
 7 strafe_right   - translate right, heading UNCHANGED
 8 fast_forward   - go straight ~6m (mid-range)
 9 super_forward  - go straight ~9m (long-range)

Key distinctions:
 - 1 vs 8 vs 9 differ ONLY in distance (short/mid/long).
 - 2/3 change heading; 6/7 keep heading. They are NOT interchangeable.


=== TASK SEMANTICS ===
You are at the CURRENT observation shown above. The history actions tell
you what has been executed so far. Your job is to choose the NEXT action
to execute from this current state.


=== REASONING RULES ===
Output EXACTLY the following 4 labeled steps inside <thinking>:

[Progress]
State which subtasks are done based on History Actions and the Current Observation.
State which subtask is active NOW.

[Observation]
Describe target object's bearing (left / right / ahead / front / behind /
up / down / above / below) AND distance (near / close / far / distant / ...).

[Reasoning]
Justify your action choice. This section MUST contain:
  (a) at least ONE direction word: left, right, ahead, front, behind,
      up, down, above, below.
  (b) at least ONE distance word: near, nearby, close, closer, far,
      distant, farther, further, short-range, mid-range, long-range,
      meter, meters.
  (c) EXACTLY ONE line in this fixed format:
      Alternative considered: <action_name> (<action_id>); Rejected because <reason>.
      where <action_name>/<action_id> must be a valid action from above,
      MUST NOT equal your final action, and the <reason> MUST itself
      contain a direction word or distance word.

[Decision]
State the final action name only (e.g., "fast_forward").


=== OUTPUT FORMAT ===
<thinking>
[Progress] ...
[Observation] ...
[Reasoning] ...
[Decision] ...
</thinking>
<next_action>X</next_action>

X must be a single integer 0-9, matching the action chosen in [Decision].
The action name in [Decision] must correspond exactly to X in <next_action>.


=== EXAMPLES ===
{example_block}


=== YOUR TASK ===
Now produce the reasoning for the current input. Follow the format exactly."""


API_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 1024,
}


# ============================================================
# 3 个示例 - curr_only 版本
# 改动:删除 Previous Observation 字段,Reasoning 不再引用前后帧对比
# ============================================================
EXAMPLE_BLOCK = '''Example 1:

[INPUT]
Overall Instruction: Proceed directly to a light beige tall building featuring multiple rectangular windows, then slightly turn right and continue heading straight towards it.

Subtasks:
1. Proceed directly to a light beige tall building featuring multiple rectangular windows.
2. Slightly turn right.
3. Continue heading straight towards it.

History Actions: [9, 9, 9, 1]

Current Subtask Hint: (empty)

Current Observation (frame 13): Currently visible is a light beige tall building featuring multiple rectangular windows, standing prominently among surrounding high-rises, with a wide street lined by trees and other modern structures nearby.

[OUTPUT]
<thinking>
[Progress] History actions [9, 9, 9, 1] cover roughly 30 meters of forward travel. The view now matches subtask 1's target — a light beige tall building with rectangular windows is prominently in front. Subtask 1 is complete. The active subtask is subtask 2: "slightly turn right".

[Observation] The target building is ahead at near-to-mid range, prominent and centered. No alternative target appears to the left. The street extends forward.

[Reasoning] Subtask 2 explicitly requires a right turn now. The target is ahead at mid-range, so another long forward move is unnecessary. A slight right rotation aligns heading for subtask 3.
Alternative considered: turn_left (2); Rejected because the target and required subtask direction are to the right, not left.

[Decision] turn_right
</thinking>
<next_action>3</next_action>


Example 2:

[INPUT]
Overall Instruction: Proceed directly ahead towards the medium-sized beige office building with rectangular windows.

Subtasks:
1. Proceed directly ahead towards the medium-sized beige office building with rectangular windows.

History Actions: []

Current Subtask Hint: (empty)

Current Observation (frame 3): Currently visible is a medium-sized beige office building with rectangular windows, standing prominently among surrounding structures, adjacent to it is a taller, dark-toned building with a grid-like facade, and in the distance, several modern skyscrapers rise against a clear blue sky.

[OUTPUT]
<thinking>
[Progress] No history actions yet. Subtask 1 is active and uncompleted: proceed ahead to the beige office building.

[Observation] The beige office building is ahead among surrounding structures. The scene shows distant background towers, so the target is at long-range.

[Reasoning] Subtask 1 requires moving straight ahead to the target. Since the target is far ahead, a long-range forward move is the best choice now.
Alternative considered: forward (1); Rejected because forward (1) is a short-range ~3m step and is insufficient for a far, long-range target.

[Decision] super_forward
</thinking>
<next_action>9</next_action>


Example 3:

[INPUT]
Overall Instruction: Move ahead to the gray office building with medium-sized rooftop equipment.

Subtasks:
1. Move ahead to the gray office building.
2. Move ahead to the gray office building with medium-sized rooftop equipment.

History Actions: [9, 8]

Current Subtask Hint: (empty)

Current Observation (frame 5): Currently visible is a gray office building featuring medium-sized rooftop equipment, including antennas and ventilation units, standing prominently against the sky with a cluster of other urban structures nearby, including a tall, modern glass-clad building to its side.

[OUTPUT]
<thinking>
[Progress] History actions [9, 8] moved the drone forward by one long-range step and one mid-range step. Both subtask 1 and subtask 2 describe reaching the same gray office building, and the drone has arrived. The remaining action is to stop at the goal.

[Observation] The target building is directly in front at close range. The rooftop equipment details are clearly visible, and no further distance needs to be covered.

[Reasoning] The task is to reach this specific building, not to pass beyond it. Since the target is close and directly in front, the correct action is to hold position.
Alternative considered: forward (1); Rejected because the target is already close in front; forward (1) is a short-range ~3m step that would carry past the reached goal.

[Decision] stop
</thinking>
<next_action>0</next_action>
'''


def build_user_prompt(
    gpt_instruction: str,
    subtask_list: str,
    history_actions: str,
    current_subtask_hint: str,
    curr_index: str,
    curr_obs: str,
    **kwargs,   # 兼容旧调用传 prev_index/prev_obs,直接忽略
) -> str:
    """填充 user prompt(curr_only 版,只用当前帧)"""
    return USER_PROMPT_TEMPLATE.format(
        gpt_instruction=gpt_instruction,
        subtask_list=subtask_list,
        history_actions=history_actions,
        current_subtask_hint=current_subtask_hint,
        curr_index=curr_index,
        curr_obs=curr_obs,
        example_block=EXAMPLE_BLOCK,
    )
