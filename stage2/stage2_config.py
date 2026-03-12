"""
Stage 2 GSPO 配置
独立配置文件，避免依赖问题
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Stage2Config:
    """Stage 2 Multi-task GSPO 配置"""
    
    # ==================== 候选生成配置 ====================
    num_candidates: int = 3
    """每个样本生成几个候选 CoT"""
    
    temperatures: List[float] = None
    """采样温度列表"""
    
    max_new_tokens: int = 300
    """生成的最大 token 数"""
    
    # ==================== GSPO 核心参数 ====================
    beta: float = 0.1
    """DPO 温度参数"""
    
    learning_rate: float = 5e-6
    """学习率"""
    
    max_grad_norm: float = 1.0
    """梯度裁剪"""
    
    # ==================== 训练流程配置 ====================
    num_rounds: int = 3
    """总共训练几轮"""
    
    steps_per_round: int = 4100
    """每轮训练多少步"""
    
    batch_size: int = 2
    """Batch size"""
    
    # ==================== 路径配置 ====================
    data_path: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/data/train_with_cot_4500.json"
    """训练数据路径"""
    
    image_base_path: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/images"
    """图片根目录"""
    
    stage1_model_path: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage1_final"
    """Stage 1 SFT 模型路径"""
    
    save_dir: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/stage2/checkpoints"
    """Stage 2 GSPO 模型保存目录"""
    
    aux_labels_path_template: str = "/home/ubuntu/data1/lyy/full_rlds_project-3/data/auxiliary_labels_round{round_num}.json"
    """辅助标签路径模板，每轮自动读取对应文件"""
    
    def get_aux_labels_path(self, round_num: int) -> str:
        """获取当前轮次的 aux_labels 文件路径"""
        return self.aux_labels_path_template.format(round_num=round_num)
    
    # ==================== 设备配置 ====================
    device: str = "cuda:0"
    """使用的 GPU 设备"""
    
    # ==================== 日志配置 ====================
    log_interval: int = 10
    """每隔多少步打印日志"""
    
    def __post_init__(self):
        """初始化后的验证和设置"""
        # 如果没有指定 temperatures，使用默认值
        if self.temperatures is None:
            self.temperatures = [0.5, 0.7, 1.0][:self.num_candidates]
        
        # 验证
        assert len(self.temperatures) == self.num_candidates, \
            f"temperatures 长度必须等于 num_candidates"
    
    def print_config(self):
        """打印配置"""
        print("=" * 60)
        print("Stage 2 GSPO 配置")
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
        print(f"  Stage 1: {self.stage1_model_path}")
        print(f"  保存: {self.save_dir}")
        print(f"\n【设备】")
        print(f"  GPU: {self.device}")
        print("=" * 60)