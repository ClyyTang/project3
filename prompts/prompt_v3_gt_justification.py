"""
Prompt v3 - GT-conditioned Justification (B+ 路线)
===================================================
任务定义:给定 target_action,让 Qwen 解释为什么这个动作在当前状态合理。
不是让 Qwen 自主决策。

核心保障:
  1. [Observation] 独立于 target_action(纯基于 obs,防循环论证)
  2. [Reasoning] 连接 observation -> target_action(解释为什么合理)
  3. 冲突词硬规则 + filter 淘汰无法自洽的样本

Created: 2026-04-16
"""

# ============================================================
# 动作信息表(每个动作的语义、冲突词、鼓励词、混淆替代集)
# ============================================================
ACTION_INFO = {
    0: {
        "name": "stop",
        "description": "stop in place, confirming the goal is reached",
        "conflict_words": ["far", "distant", "long-range", "long range",
                           "continue forward", "proceed toward", "proceed towards",
                           "still need to approach", "not yet close"],
        "encouraged_words": ["reached", "arrived", "goal is confirmed",
                             "no further approach needed", "close", "in front"],
        "confusion_set": [1, 8],
    },
    1: {
        "name": "forward",
        "description": "go straight ~3m (short-range step)",
        "conflict_words": ["reached", "arrived", "already at", "goal is reached",
                           "stop here", "very far", "long-range", "long range"],
        "encouraged_words": ["close", "near", "short-range", "short range",
                             "small remaining distance", "almost there"],
        "confusion_set": [8, 9],
    },
    2: {
        "name": "turn_left",
        "description": "rotate left in place (heading changes, no translation)",
        "conflict_words": [],  # soft only for turn
        "encouraged_words": ["left", "to the left", "lateral"],
        "confusion_set": [3],
    },
    3: {
        "name": "turn_right",
        "description": "rotate right in place (heading changes, no translation)",
        "conflict_words": [],
        "encouraged_words": ["right", "to the right", "lateral"],
        "confusion_set": [2],
    },
    4: {
        "name": "ascend",
        "description": "move upward",
        "conflict_words": ["descend", "downward", "moving down", "go down", "lower"],
        "encouraged_words": ["above", "upward", "higher", "rise", "up"],
        "confusion_set": [5],
    },
    5: {
        "name": "descend",
        "description": "move downward",
        "conflict_words": ["ascend", "upward", "moving up", "go up", "rise", "higher"],
        "encouraged_words": ["below", "downward", "lower", "drop", "down"],
        "confusion_set": [4],
    },
    6: {
        "name": "strafe_left",
        "description": "translate left, heading UNCHANGED",
        "conflict_words": [],
        "encouraged_words": ["left"],
        "confusion_set": [7, 2],
    },
    7: {
        "name": "strafe_right",
        "description": "translate right, heading UNCHANGED",
        "conflict_words": [],
        "encouraged_words": ["right"],
        "confusion_set": [6, 3],
    },
    8: {
        "name": "fast_forward",
        "description": "go straight ~6m (mid-range step)",
        "conflict_words": ["reached", "arrived", "already at", "goal is reached",
                           "very close", "stop now"],
        "encouraged_words": ["mid-range", "mid range", "moderate distance",
                             "not yet close", "some way off"],
        "confusion_set": [1, 9],
    },
    9: {
        "name": "super_forward",
        "description": "go straight ~9m (long-range step)",
        "conflict_words": ["reached", "arrived", "already at", "goal is reached",
                           "very close", "stop now"],
        "encouraged_words": ["far", "distant", "long-range", "long range",
                             "still some way off", "significant distance"],
        "confusion_set": [8, 1],
    },
}

ACTION_NAMES = {k: v["name"] for k, v in ACTION_INFO.items()}
NAME_TO_ID = {v: k for k, v in ACTION_NAMES.items()}


SYSTEM_PROMPT = """You are a drone navigation reasoning assistant.

You will be given a PRE-DETERMINED correct action. Your task is to produce
reasoning that JUSTIFIES why this action is the appropriate choice for the
current observation and navigation state.

You are NOT making a decision. You are EXPLAINING a given decision.
You MUST justify the pre-determined action and MUST NOT output a different
final action.

CRITICAL RULES:
1. Your [Observation] section must describe what you ACTUALLY see, based
   ONLY on the observation text (and image if provided). Do NOT let the
   pre-determined action influence what you claim to observe. Do NOT
   mention the action name or action number in [Observation].
2. Your [Reasoning] section must then connect your observation to the
   pre-determined action, explaining why it is appropriate.
3. Do not copy wording from examples; adapt to the current input."""


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


=== PRE-DETERMINED ACTION ===
The correct next action is: {target_action_id} ({target_action_name})
Meaning: {target_action_description}

You MUST justify this action. Your <next_action> MUST be {target_action_id}.


=== ACTION SPACE (for reference) ===
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


=== CONSTRAINT ===
In your [Reasoning] section, the following words are FORBIDDEN
(they contradict the pre-determined action):
{conflict_words_display}

The following words are ENCOURAGED (they support the action):
{encouraged_words_display}


=== REASONING RULES ===
Output EXACTLY the following 4 labeled steps inside <thinking>:

[Progress]
State which subtasks are done based on History Actions and the Current
Observation. State which subtask is active NOW.

[Observation]
Describe what you observe: target object's bearing (left / right / ahead /
front / behind / up / down / above / below) AND distance (near / close /
far / distant / ...).
IMPORTANT: This section must be based ONLY on the observation text.
Do NOT mention any action name or action number here.

[Reasoning]
Explain why the pre-determined action is appropriate given your observation.
This section MUST contain:
  (a) at least ONE direction word: left, right, ahead, front, behind,
      up, down, above, below.
  (b) at least ONE distance word: near, nearby, close, closer, far,
      distant, farther, further, short-range, mid-range, long-range,
      meter, meters.
  (c) EXACTLY ONE line in this fixed format:
      Alternative considered: <action_name> (<action_id>); Rejected because <reason>.
      The alternative MUST come from this set: {confusion_set_display}
      The <reason> MUST contain a direction word or distance word.

[Decision]
State the final action name: {target_action_name}


=== OUTPUT FORMAT ===
<thinking>
[Progress] ...
[Observation] ...
[Reasoning] ...
[Decision] ...
</thinking>
<next_action>{target_action_id}</next_action>


=== EXAMPLES ===
{example_block}


=== YOUR TASK ===
Now produce the justification for the pre-determined action. Follow the format exactly."""


API_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 1024,
}


# ============================================================
# 5 个示例(justification 模式)
# ============================================================
EXAMPLE_BLOCK = '''Example 1 (justifying turn_right):

[INPUT]
Pre-determined action: 3 (turn_right)
Current Observation: A light beige tall building with rectangular windows stands prominently among surrounding high-rises, with a wide street lined by trees.
History: [9, 9, 9, 1]
Active subtask: "Slightly turn right."

[OUTPUT]
<thinking>
[Progress] History actions [9, 9, 9, 1] have brought the drone forward. The beige building from subtask 1 is now in view, so subtask 1 is complete. The active subtask is subtask 2: "slightly turn right."

[Observation] The beige building is ahead at near-to-mid range, prominent and centered. The street extends forward with no visible obstruction.

[Reasoning] The active subtask explicitly requires a right heading change. The target is ahead at mid-range, so the drone does not need more forward distance now; instead it needs to adjust heading to align with the next waypoint. A right rotation satisfies the subtask requirement.
Alternative considered: turn_left (2); Rejected because the subtask specifies right, not left, and the next target is to the right of the current heading.

[Decision] turn_right
</thinking>
<next_action>3</next_action>


Example 2 (justifying super_forward — target is far):

[INPUT]
Pre-determined action: 9 (super_forward, ~9m)
Current Observation: A medium-sized beige office building with rectangular windows stands among surrounding structures, with distant skyscrapers in the background against a clear sky.
History: []
Active subtask: "Proceed directly ahead towards the beige office building."

[OUTPUT]
<thinking>
[Progress] No history actions yet. Subtask 1 is active: proceed ahead to the beige office building.

[Observation] The beige office building is ahead among surrounding structures. The background still shows distant towers, and the target appears at long-range, occupying a small portion of the view.

[Reasoning] The target is far ahead with significant distance remaining. The subtask requires approaching it, and heading is already aligned. A long-range step is appropriate to make efficient progress toward the distant target.
Alternative considered: fast_forward (8); Rejected because the target is still at long-range and a mid-range 6m step would be insufficient progress.

[Decision] super_forward
</thinking>
<next_action>9</next_action>


Example 3 (justifying fast_forward — target is at mid-range):

[INPUT]
Pre-determined action: 8 (fast_forward, ~6m)
Current Observation: A cluster of modern office buildings ahead with dark-glass facades, occupying a noticeable portion of the front view, with a wide street extending between them.
History: [9]
Active subtask: "Go straight to reach the cluster of modern office buildings."

[OUTPUT]
<thinking>
[Progress] History [9] covered about 9m. Subtask 1 is still active.

[Observation] The cluster is directly ahead, occupying a noticeable portion of the view. The street extends between the buildings. The target is at mid-range, not yet close enough to fill the frame.

[Reasoning] The target is ahead at mid-range with moderate remaining distance. A mid-range step matches the observed distance appropriately, closing the gap without risking overshoot.
Alternative considered: super_forward (9); Rejected because the target already occupies a noticeable portion ahead; a long-range 9m step would likely overshoot.

[Decision] fast_forward
</thinking>
<next_action>8</next_action>


Example 4 (justifying forward — target is close):

[INPUT]
Pre-determined action: 1 (forward, ~3m)
Current Observation: The dark brown residential tower with white balcony railings stands close and prominent in the center of the view, occupying a large portion of the frame.
History: [9, 8]
Active subtask: "Go straight to reach the dark brown residential tower."

[OUTPUT]
<thinking>
[Progress] History [9, 8] covered about 15m. Subtask 1 is still active.

[Observation] The dark brown tower is ahead, prominent and centered, occupying a large portion of the frame. The balcony railings are clearly visible, indicating close range.

[Reasoning] The target is ahead at close range with only a small remaining distance. A short-range step is appropriate to complete the approach without overshooting.
Alternative considered: fast_forward (8); Rejected because the target is already close; a mid-range 6m step would overshoot.

[Decision] forward
</thinking>
<next_action>1</next_action>


Example 5 (justifying stop — goal is reached):

[INPUT]
Pre-determined action: 0 (stop)
Current Observation: A gray office building with medium-sized rooftop equipment including antennas and ventilation units stands prominently against the sky, clearly filling much of the view.
History: [9, 8]
Active subtask: "Move ahead to the gray office building with rooftop equipment."

[OUTPUT]
<thinking>
[Progress] History [9, 8] covered about 15m. The gray office building with rooftop equipment is now clearly in view and the drone has arrived. All subtasks are complete.

[Observation] The target building is directly in front, filling much of the view. Rooftop equipment details are clearly visible, confirming close proximity.

[Reasoning] The target fills the view and all details are resolvable, confirming the drone has reached the goal. No further forward progress is needed; holding position completes the task.
Alternative considered: forward (1); Rejected because the target is already close in front; a further short-range step would carry past the reached goal.

[Decision] stop
</thinking>
<next_action>0</next_action>
'''


# ============================================================
# build_user_prompt
# ============================================================
def build_user_prompt(
    gpt_instruction: str,
    subtask_list: str,
    history_actions: str,
    current_subtask_hint: str,
    curr_index: str,
    curr_obs: str,
    target_action: int,
    **kwargs,
) -> str:
    """
    填充 user prompt。
    target_action 是归一化后的 GT 动作(0-9)。
    """
    info = ACTION_INFO.get(target_action)
    if info is None:
        raise ValueError(f"unknown target_action: {target_action}")

    # 冲突词显示
    if info["conflict_words"]:
        conflict_display = "  " + ", ".join(f'"{w}"' for w in info["conflict_words"])
    else:
        conflict_display = "  (none for this action)"

    # 鼓励词显示
    if info["encouraged_words"]:
        encouraged_display = "  " + ", ".join(f'"{w}"' for w in info["encouraged_words"])
    else:
        encouraged_display = "  (none)"

    # 混淆替代集显示
    confusion_set = info["confusion_set"]
    confusion_display = ", ".join(
        f"{ACTION_INFO[a]['name']} ({a})" for a in confusion_set
    )

    return USER_PROMPT_TEMPLATE.format(
        gpt_instruction=gpt_instruction,
        subtask_list=subtask_list,
        history_actions=history_actions,
        current_subtask_hint=current_subtask_hint,
        curr_index=curr_index,
        curr_obs=curr_obs,
        target_action_id=target_action,
        target_action_name=info["name"],
        target_action_description=info["description"],
        conflict_words_display=conflict_display,
        encouraged_words_display=encouraged_display,
        confusion_set_display=confusion_display,
        example_block=EXAMPLE_BLOCK,
    )
