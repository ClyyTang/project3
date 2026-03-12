"""
GSPO 配置类
定义所有超参数和路径配置
"""
from dataclasses import dataclass
from typing import List


@dataclass
class GSPOConfig:
    """
    GSPO (Group Relative Policy Optimization) 配置
    
    核心思想：
    - 为每个样本生成多个候选 CoT
    - 根据 reward 选择最好和最差的
    - 用 DPO loss 优化：让模型偏好好的，远离差的
    """
    
    # ==================== 候选生成配置 ====================
    
    num_candidates: int = 2
    """每个样本生成几个候选 CoT"""
    
    temperatures: List[float] = None
    """采样温度列表（长度应等于 num_candidates）"""
    
    max_new_tokens: int = 300
    """生成的最大 token 数"""
    
    # ==================== GSPO 核心参数 ====================
    
    beta: float = 0.1
    """DPO 温度参数，控制偏好强度（越大惩罚越强）"""
    
    learning_rate: float = 5e-6
    """学习率（比 SFT 低一个数量级）"""
    
    max_grad_norm: float = 1.0
    """梯度裁剪"""
    
    # ==================== 训练流程配置 ====================
    
    num_rounds: int = 3
    """总共训练几轮（每轮重新生成候选）"""
    
    steps_per_round: int = 4100
    """每轮训练多少步"""
    
    batch_size: int = 1
    """训练时的 batch size（建议 >=2）"""
    
    # ==================== 路径配置 ====================
    
    
    data_path: str = "/home/ubuntu/data1/lyy/full_rlds_project/2_data_merge/train_with_cot_1500.json"
    """训练数据路径"""

    image_base_path: str = "/home/ubuntu/data1/lyy/full_rlds_project/images"
    """图片根目录"""
    
    base_checkpoint: str = "/home/ubuntu/data1/zx/OpenFly-Platform/train/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
    """VLA 基础模型 checkpoint"""
    
    stage1_model_path: str = "/home/ubuntu/data1/lyy/full_rlds_project/3_training/checkpoints/stage1_sft"
    """Stage 1 SFT 模型路径（LoRA + projector）"""
    
    save_dir: str = "/home/ubuntu/data1/lyy/full_rlds_project/3_training/checkpoints/stage2_gspo"
    """Stage 2 GSPO 模型保存目录"""
    
    # ==================== 设备配置 ====================
    
    device: str = "cuda:7"
    """使用的 GPU 设备"""
    
    # ==================== 日志配置 ====================
    
    log_interval: int = 10
    """每隔多少步打印日志"""
    
    save_candidates: bool = True
    """是否保存每轮生成的候选数据（调试用）"""
    
    def __post_init__(self):
        """初始化后的验证和设置"""
        
        # 如果没有指定 temperatures，使用默认值
        if self.temperatures is None:
            self.temperatures = [0.7, 1.0, 1.5]
        
        # 验证
        assert len(self.temperatures) == self.num_candidates, \
            f"temperatures 长度 ({len(self.temperatures)}) 必须等于 num_candidates ({self.num_candidates})"
        
        assert self.beta > 0, "beta 必须 > 0"
        assert self.learning_rate > 0, "learning_rate 必须 > 0"
        assert self.num_rounds > 0, "num_rounds 必须 > 0"
        assert self.steps_per_round > 0, "steps_per_round 必须 > 0"
        assert self.batch_size > 0, "batch_size 必须 > 0"
        
        # batch_size 建议 >= 2（否则 batch norm 等可能有问题）
        if self.batch_size == 2:
            print("⚠️  警告: batch_size=1 可能导致训练不稳定")
    
    def print_config(self):
        """打印配置（用于训练开始时）"""
        print("=" * 60)
        print("GSPO 配置")
        print("=" * 60)
        print(f"\n【候选生成】")
        print(f"  候选数量: {self.num_candidates}")
        print(f"  采样温度: {self.temperatures}")
        print(f"  最大 tokens: {self.max_new_tokens}")
        print(f"\n【GSPO 参数】")
        print(f"  Beta (DPO温度): {self.beta}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  梯度裁剪: {self.max_grad_norm}")
        print(f"\n【训练流程】")
        print(f"  总轮数: {self.num_rounds}")
        print(f"  每轮步数: {self.steps_per_round}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  总训练步数: {self.num_rounds * self.steps_per_round}")
        print(f"\n【路径】")
        print(f"  数据: {self.data_path}")
        print(f"  Stage 1 模型: {self.stage1_model_path}")
        print(f"  保存目录: {self.save_dir}")
        print(f"\n【设备】")
        print(f"  GPU: {self.device}")
        print("=" * 60)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试配置类"""
    
    print("测试 GSPOConfig\n")
    
    # 1. 默认配置
    config = GSPOConfig()
    config.print_config()
    
    # 2. 自定义配置
    print("\n\n测试自定义配置:\n")
    custom_config = GSPOConfig(
        num_candidates=5,
        temperatures=[0.5, 0.7, 1.0, 1.3, 1.5],
        beta=0.2,
        num_rounds=2,
        batch_size=8,
        device="cuda:1"
    )
    custom_config.print_config()
    
    # 3. 错误配置（应该报错）
    print("\n\n测试错误配置:")
    try:
        bad_config = GSPOConfig(
            num_candidates=3,
            temperatures=[0.7, 1.0]  # 长度不匹配
        )
    except AssertionError as e:
        print(f"✅ 捕获到预期错误: {e}")
    
    print("\n✅ GSPOConfig 测试通过!")