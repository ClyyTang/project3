"""
动作语义统一定义 — 全项目共享
所有 prompt 生成、数据过滤、训练/eval 都从这里 import
"""

ACTION_SEMANTICS = {
    0: {"name": "stop",          "desc_en": "stop in place",                     "desc_cn": "停止"},
    1: {"name": "forward",       "desc_en": "go straight ~3m (short)",           "desc_cn": "直行短距(3米)"},
    2: {"name": "turn_left",     "desc_en": "rotate left in place",              "desc_cn": "原地左转"},
    3: {"name": "turn_right",    "desc_en": "rotate right in place",             "desc_cn": "原地右转"},
    4: {"name": "ascend",        "desc_en": "move upward",                       "desc_cn": "上升"},
    5: {"name": "descend",       "desc_en": "move downward",                     "desc_cn": "下降"},
    6: {"name": "strafe_left",   "desc_en": "translate left, heading unchanged", "desc_cn": "左平移(朝向不变)"},
    7: {"name": "strafe_right",  "desc_en": "translate right, heading unchanged","desc_cn": "右平移(朝向不变)"},
    8: {"name": "fast_forward",  "desc_en": "go straight ~6m (medium)",          "desc_cn": "直行中距(6米)"},
    9: {"name": "super_forward", "desc_en": "go straight ~9m (long)",            "desc_cn": "直行长距(9米)"},
}

ALIAS_MAP = {-1: 4, -2: 5}

STRICT_ACTIONS = {0, 2, 3, 4, 5, 6, 7}
FORWARD_FAMILY = {1, 8, 9}

def normalize_action(a: int) -> int:
    """把 -1/-2 映射成 4/5,其他保持不变"""
    return ALIAS_MAP.get(a, a)

def actions_match(predicted: int, gt: int, mode: str = "mixed") -> bool:
    """
    判断 Qwen 推理的动作是否可以接受作为该样本的 GT 配对。
    
    mode:
      - "strict": 必须完全一致才返回 True
      - "mixed":  完全一致 OR 同属 FORWARD_FAMILY (1/8/9)
    
    GT action 永远使用原始值,这里只用于过滤 thinking 是否保留。
    """
    if mode not in {"strict", "mixed"}:
        raise ValueError(f"Unsupported mode: {mode!r}, must be 'strict' or 'mixed'")
    p, g = normalize_action(predicted), normalize_action(gt)
    if p == g:
        return True
    if mode == "mixed" and p in FORWARD_FAMILY and g in FORWARD_FAMILY:
        return True
    return False
