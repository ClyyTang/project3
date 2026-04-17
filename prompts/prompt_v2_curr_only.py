"""
Prompt v2 - curr_only 版本(基于 v1 加 4 条保护)
===============================================
改动相对 v1:
  (1) ACTION FAMILY FIRST  : 先判动作族,再判距离
  (2) DISTANCE DECISION GUIDE: 前进族才适用,且"朝向已对齐 + subtask 仍需接近"才偏长
  (3) ANTI-BIAS RULE       : 压 pred=3 偏置(仅当 subtask 明说转 或有侧向线索才转)
  (4) 示例扩到 5 个        : 新增 forward(1) 和 fast_forward(8),三档各有正例

Created: 2026-04-16
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

Do not default to any specific action when uncertain. Do not copy wording
from examples; adapt to the current input."""


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


=== ACTION FAMILY FIRST (read before anything else) ===

Before choosing any distance (1/8/9), first determine the action family
required by the ACTIVE subtask:

- If the active subtask explicitly requires "turn" / "rotate" / "face" /
  mentions heading change → the family is {{turn_left(2), turn_right(3)}}.
  Use the DISTANCE GUIDE only AFTER turning is done.

- If the active subtask explicitly requires "ascend" / "go up" / "rise" /
  "descend" / "go down" / "lower" → the family is {{ascend(4), descend(5)}}.
  Do NOT switch to forward distances because the target looks "close".

- If the active subtask requires moving to a landmark (most common case)
  AND heading is already aligned with that landmark (the landmark is
  ahead/front-center, not to the side) → the family is forward, and the
  DISTANCE GUIDE applies.

- Only after all forward progress is confirmed complete AND the final
  landmark is reached, consider stop(0).


=== DISTANCE DECISION GUIDE (applies ONLY to forward family) ===

Use this guide ONLY AFTER ACTION FAMILY FIRST has concluded the family is
forward AND the heading is aligned with the target (target appears ahead,
front-center, not to one side).

Choose the distance by how the target appears in the current observation:

- Target occupies MOST of the view (very close, centered, fills much of
  frame) AND no remaining subtask requires further approach:
  choose 0 (stop).

- Target clearly visible at CLOSE range (near-to-mid, prominent but not
  filling view) AND active subtask still asks to approach:
  choose 1 (forward, ~3m, short step).

- Target visible at MID range (recognizable but still some way off,
  occupies a noticeable portion of the view):
  choose 8 (fast_forward, ~6m).

- Target visible but at LONG range (distant, small in view, scene still
  shows significant background around the target):
  choose 9 (super_forward, ~9m).

PROGRESS BIAS: when the active subtask still requires clear forward
progress AND heading is aligned AND the target is not yet close, prefer
the LONGER step (9 over 8, 8 over 1). Stopping (0) requires that the
goal is REACHED, not merely VISIBLE.


=== ANTI-BIAS RULE (for turn_left / turn_right) ===

Do NOT choose turn_left (2) or turn_right (3) unless:
  (a) the active subtask explicitly requires a turn (mentions
      "turn"/"rotate"/"face"/heading change), OR
  (b) the current observation has a clear lateral cue, i.e., the target
      is visibly to the left or right of the forward direction.

If neither (a) nor (b) applies, do NOT turn; choose from the forward
family or the vertical family instead.


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
# 5 个示例(v2: 三档前进 1/8/9 各一个 + turn_right + stop)
# 每个示例的 Alternative 都经过审查:来自混淆族、含方向或距离词
# ============================================================
EXAMPLE_BLOCK = '''Example 1 (turn_right, triggered by explicit subtask):

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
[Progress] History actions [9, 9, 9, 1] cover roughly 30 meters of forward travel. The light beige tall building described in subtask 1 is now prominently in front. Subtask 1 is complete. The active subtask is subtask 2: "slightly turn right".

[Observation] The target building is ahead at near-to-mid range, prominent and centered. The street extends forward.

[Reasoning] The active subtask EXPLICITLY requires a right turn, satisfying the anti-bias rule. The target is ahead at mid-range, so another long forward move is unnecessary right now. A slight right rotation aligns heading for subtask 3.
Alternative considered: turn_left (2); Rejected because the subtask direction is to the right, not left.

[Decision] turn_right
</thinking>
<next_action>3</next_action>


Example 2 (super_forward, long range, heading aligned):

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

[Observation] The target beige building is ahead, among surrounding structures. The background shows additional distant towers, indicating the scene still extends far forward. The target is at long-range.

[Reasoning] Family is forward because the subtask requires approaching a landmark and heading is aligned (target ahead, not to the side). The target is far, long-range, and progress is still needed, so the progress bias favors the longest step.
Alternative considered: fast_forward (8); Rejected because fast_forward is a mid-range ~6m step that is insufficient for a long-range target.

[Decision] super_forward
</thinking>
<next_action>9</next_action>


Example 3 (fast_forward, mid-range, heading aligned):

[INPUT]
Overall Instruction: Go straight to reach a cluster of modern office buildings ahead.

Subtasks:
1. Go straight to reach the cluster of modern office buildings ahead.

History Actions: [9]

Current Subtask Hint: (empty)

Current Observation (frame 2): Currently visible is a cluster of modern office buildings ahead, with dark-glass facades and rectangular window grids, occupying a noticeable portion of the front view; a wide street extends between them.

[OUTPUT]
<thinking>
[Progress] History action [9] has already covered about 9 meters. Subtask 1 is still active — reaching the cluster of modern office buildings.

[Observation] The cluster is directly ahead, occupies a noticeable portion of the view, and the street clearly extends between the buildings. The target is at mid-range, not yet close enough to fill the frame.

[Reasoning] Family is forward because heading is aligned with the cluster and the subtask requires further approach. The target is at mid-range (recognizable but with remaining distance), which matches the fast_forward regime.
Alternative considered: super_forward (9); Rejected because the target already occupies a noticeable portion ahead, so a long-range 9m step would likely overshoot.

[Decision] fast_forward
</thinking>
<next_action>8</next_action>


Example 4 (forward, close range, heading aligned):

[INPUT]
Overall Instruction: Go straight to reach the dark brown residential tower with white balcony railings.

Subtasks:
1. Go straight to reach the dark brown residential tower with white balcony railings.

History Actions: [9, 8]

Current Subtask Hint: (empty)

Current Observation (frame 4): Currently visible is the dark brown residential tower with white balcony railings, standing close and prominent in the center of the view, occupying a large portion of the frame, with other buildings partially visible to the sides.

[OUTPUT]
<thinking>
[Progress] History actions [9, 8] cover roughly 15 meters of forward travel, bringing the drone close to the target. Subtask 1 is still active — the tower is visible but the drone has not yet reached it.

[Observation] The dark brown tower is ahead at close range, prominent and centered. The balcony railings are clearly resolvable, indicating near-to-mid distance.

[Reasoning] Family is forward because heading is aligned (target ahead) and a small remaining approach is still needed. At close range, a short-range step is appropriate to avoid overshoot.
Alternative considered: fast_forward (8); Rejected because fast_forward is a mid-range ~6m step that would likely overshoot the close target.

[Decision] forward
</thinking>
<next_action>1</next_action>


Example 5 (stop, reached):

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
[Progress] History actions [9, 8] moved the drone forward by one long-range step and one mid-range step. Both subtask 1 and subtask 2 describe reaching the same gray office building, and the rooftop equipment is now clearly resolvable. The drone has arrived.

[Observation] The target building is directly in front at close range. The rooftop equipment details are clearly visible, and no further forward progress is required.

[Reasoning] The task is to REACH this building, and the goal is now reached (target fills the view, no remaining approach). Stopping is appropriate because the goal is confirmed reached, not merely visible.
Alternative considered: forward (1); Rejected because the target is already close in front; a further short-range step would carry past the reached goal.

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
    **kwargs,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        gpt_instruction=gpt_instruction,
        subtask_list=subtask_list,
        history_actions=history_actions,
        current_subtask_hint=current_subtask_hint,
        curr_index=curr_index,
        curr_obs=curr_obs,
        example_block=EXAMPLE_BLOCK,
    )
