"""
Stage 3 Risk-Aware GSPO 配置
基于 stage2_config.py 扩展，新增 Risk Predictor 相关超参数
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Stage3Config:
    """Stage 3 Risk-Aware GSPO 配置"""

    # ==================== 候选生成配置 ====================
    num_candidates: int = 3
    temperatures: List[float] = None
    max_new_tokens: int = 300

    # ==================== GSPO 核心参数 ====================
    beta: float = 0.1
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0

    # ==================== 训练流程配置 ====================
    num_rounds: int = 1
    steps_per_round: int = 19000
    batch_size: int = 1

    # ==================== 路径配置 ====================
    data_path: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/data/train_with_cot_4500.json"

    # ⭐ 新增：aux_labels 路径模板，每轮自动读取对应文件
    # 使用时：aux_labels_path_template.format(round_num=0)
    aux_labels_path_template: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/data/auxiliary_labels_round{round_num}.json"

    image_base_path: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/images"

    # ⭐ 改为从 stage2 权重开始训练
    stage2_model_path: str = '/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage2_final'

    save_dir: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage3_final"

    # ==================== 设备配置 ====================
    device: str = "cuda:0"

    # ==================== 日志配置 ====================
    log_interval: int = 10

    # ==================== ⭐ Stage 3 新增：Risk 超参数 ====================

    # risk_loss 在 total_loss 中的整体权重
    # total_loss = gspo_loss + aux_loss + alpha * risk_loss
    alpha: float = 0.5

    # risk_loss_1 和 risk_loss_2 之间的平衡系数
    # risk_loss = risk_loss_1 + gamma * risk_loss_2
    # risk_loss_1 用全部1500样本训，risk_loss_2 只用454弱样本训
    # 数据量差约3倍，gamma 优先调整
    gamma: float = 1.0

    # 风险加权 loss 的系数
    # risk_weight = 1.0 + mu * overall_risk
    # mu=0.5 意味着最高风险样本的 loss 权重是普通样本的 1.5 倍
    mu: float = 0.5

    # risk_loss warmup 步数，前 warmup_steps 步 alpha 从0线性增长到设定值
    # 防止随机初始化的 risk heads 在训练初期干扰主模型
    warmup_steps: int = 200

    # risk_loss warmup 步数，前 warmup_steps 步 alpha 从0线性增长到设定值
    # 防止随机初始化的 risk heads 在训练初期干扰主模型
    warmup_steps: int = 200

    # ==================== ⭐ Stage 3 新增：推理阶段风险阈值 ====================
    # 训练时不用，推理阶段分级策略用
    risk_threshold_green: float = 0.3   # 低于此值：绿色，正常生成
    risk_threshold_red: float = 0.7     # 高于此值：红色，激活 Correction LoRA

    # ==================== ⭐ error_type 映射 ====================
    error_type_map: dict = field(default_factory=lambda: {
        "perception": 0,
        "comprehension": 1,
        "reasoning": 2,
        "decision": 3
    })

    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.5, 0.7, 1.0][:self.num_candidates]

        assert len(self.temperatures) == self.num_candidates, \
            "temperatures 长度必须等于 num_candidates"

        assert self.risk_threshold_green < self.risk_threshold_red, \
            "green阈值必须小于red阈值"

    def get_aux_labels_path(self, round_num: int) -> str:
        """获取当前轮次的 aux_labels 文件路径"""
        return self.aux_labels_path_template.format(round_num=round_num)

    def print_config(self):
        print("=" * 60)
        print("Stage 3 Risk-Aware GSPO 配置")
        print("=" * 60)
        print(f"\n【候选生成】")
        print(f"  候选数量: {self.num_candidates}")
        print(f"  采样温度: {self.temperatures}")
        print(f"  最大 tokens: {self.max_new_tokens}")
        print(f"\n【GSPO 参数】")
        print(f"  Beta: {self.beta}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  梯度裁剪: {self.max_grad_norm}")
        print(f"\n【训练流程】")
        print(f"  总轮数: {self.num_rounds}")
        print(f"  每轮步数: {self.steps_per_round}")
        print(f"  Batch size: {self.batch_size}")
        print(f"\n【路径】")
        print(f"  数据: {self.data_path}")
        print(f"  aux_labels模板: {self.aux_labels_path_template}")
        print(f"  Stage2权重: {self.stage2_model_path}")
        print(f"  保存: {self.save_dir}")
        print(f"\n【Risk 超参数】")
        print(f"  alpha (risk_loss权重): {self.alpha}")
        print(f"  gamma (loss_1/loss_2平衡): {self.gamma}")
        print(f"  mu (风险加权系数): {self.mu}")
        print(f"\n【推理阈值】")
        print(f"  绿色 < {self.risk_threshold_green} < 黄色 < {self.risk_threshold_red} < 红色")
        print(f"\n【error_type 映射】")
        for k, v in self.error_type_map.items():
            print(f"  {k} → {v}")
        print("=" * 60)
