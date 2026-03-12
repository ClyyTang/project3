"""
Risk-Aware VLA 模型 (RiskAwareVLA)

基于 stage2/multitask_model.py 扩展，新增：
- risk_head_1: 整体风险度预测（标量回归，0~1）
- risk_head_2: 错误类型预测（4分类）

加载顺序：
1. 加载 base VLA（openfly-agent-7b）
2. 加载 stage2 LoRA 权重
3. 加载 stage2 projector
4. 加载 stage2 auxiliary_heads（4个辅助头）
5. 新增 risk_head_1 和 risk_head_2（随机初始化，待训练）
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

# 添加OpenFly-Platform到路径
OPENFLY_PATH = '/home/ubuntu/data1/lyy/OpenFly-Platform'
if OPENFLY_PATH not in sys.path:
    sys.path.insert(0, os.path.join(OPENFLY_PATH, 'train'))

try:
    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoImageProcessor, AutoProcessor
    from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

    OPENVLA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入OpenVLA模块: {e}")
    OPENVLA_AVAILABLE = False
    OpenVLAForActionPrediction = None


class RiskAwareVLA(nn.Module):
    """
    Risk-Aware VLA 模型

    架构：
    - base_vla: Stage 2 训练的 VLA 模型
    - keyword_head:        关键词识别（34维）    ← 从stage2加载
    - direction_head:      方向分类（4维）       ← 从stage2加载
    - cot_quality_head:    CoT质量评分（1维）    ← 从stage2加载
    - action_validity_head:动作合理性（1维）     ← 从stage2加载
    - risk_head_1:         整体风险度（1维）     ← 新增，随机初始化
    - risk_head_2:         错误类型预测（4维）   ← 新增，随机初始化
    """

    def __init__(
        self,
        stage2_checkpoint_dir: str,
        num_unfrozen_layers: int = 4,
        num_keywords: int = 34,
        hidden_dim: int = 4096,
        dropout: float = 0.1,
        verbose: bool = True
    ):
        """
        Args:
            stage2_checkpoint_dir: Stage 2 checkpoint目录路径
            num_unfrozen_layers: 解冻的LLM层数（从后往前）
            num_keywords: 关键词总数
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
            verbose: 是否打印详细信息
        """
        super().__init__()

        self.verbose = verbose
        self.num_unfrozen_layers = num_unfrozen_layers
        self.num_keywords = num_keywords
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if self.verbose:
            print(f"\n{'='*60}")
            print("初始化 RiskAwareVLA 模型")
            print(f"{'='*60}")

        if not OPENVLA_AVAILABLE:
            raise RuntimeError("OpenVLA模块不可用，请检查OpenFly-Platform路径")

        # 1. 加载 stage2 模型（base_vla + LoRA + projector）
        if self.verbose:
            print(f"加载 Stage 2 模型: {stage2_checkpoint_dir}")
        self.base_vla = self._load_stage2_model(stage2_checkpoint_dir)

        # 2. 设置冻结策略（和stage2一致）
        if self.verbose:
            print(f"\n设置冻结策略: 解冻最后{num_unfrozen_layers}层")
        self._setup_freezing_strategy(num_unfrozen_layers)

        # 3. 初始化所有辅助heads（4个旧的 + 2个新的）
        if self.verbose:
            print(f"\n初始化辅助heads（含新增risk heads）")
        self._setup_auxiliary_heads()

        # 4. 加载stage2的4个辅助heads权重
        aux_heads_path = Path(stage2_checkpoint_dir) / "auxiliary_heads.pt"
        if aux_heads_path.exists():
            if self.verbose:
                print(f"\n加载 stage2 auxiliary_heads: {aux_heads_path}")
            self._load_auxiliary_heads(aux_heads_path)
        else:
            if self.verbose:
                print(f"\n⚠️  未找到 auxiliary_heads.pt，4个辅助头使用随机初始化")

        # 5. 设置 hidden state hook
        if self.verbose:
            print(f"\n设置 hidden state hook")
        self._setup_hidden_state_hook()

        # 6. 打印参数统计
        if self.verbose:
            self._print_parameter_stats()
            print(f"\n{'='*60}")
            print("✅ RiskAwareVLA 初始化完成")
            print(f"{'='*60}\n")

        self._debug_count = 0

    def _load_stage2_model(self, checkpoint_dir: str):
        """加载 Stage 2 训练的 VLA 模型"""
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")

        try:
            # ===== Monkey patch =====
            import model.llm_backbone as llm_backbone_module
            from transformers import LlamaForCausalLM, AutoConfig

            def patched_init(self, llm_max_length, hf_token=None,
                             use_flash_attention_2=False, inference_mode=False):
                super(llm_backbone_module.LLaMa2LLMBackbone, self).__init__()
                self.llm_max_length = llm_max_length
                self.inference_mode = inference_mode

                hf_hub_path = "/home/ubuntu/data1/zx/OpenFly-Platform/train/meta-llama/llama2-7b-hf"

                if not inference_mode:
                    self.llm = LlamaForCausalLM.from_pretrained(
                        hf_hub_path, local_files_only=True,
                        do_sample=False, temperature=1.0, top_p=1.0
                    )
                else:
                    llm_config = AutoConfig.from_pretrained(
                        hf_hub_path, local_files_only=True)
                    self.llm = LlamaForCausalLM._from_config(llm_config)

                self.tokenizer = llm_backbone_module.AutoTokenizer.from_pretrained(
                    hf_hub_path, local_files_only=True,
                    model_max_length=llm_max_length,
                    padding_side="right", use_fast=False
                )
                self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
                self.llm.resize_token_embeddings(
                    len(self.tokenizer), pad_to_multiple_of=64)

            llm_backbone_module.LLaMa2LLMBackbone.__init__ = patched_init

            # ===== 初始化 overwatch =====
            from model.overwatch import initialize_overwatch
            overwatch = initialize_overwatch("risk-aware-vla")

            # ===== 加载 base VLA =====
            base_checkpoint_dir = "/home/ubuntu/data1/lyy/openfly-agent-7b"

            from model.load_model import OpenFly
            from model.vision_backbone import DinoSigLIPViTBackbone
            from model.action_tokenizer import ActionTokenizer
            from model.llm_backbone import LLaMa2LLMBackbone
            from safetensors.torch import load_file

            dataset_statistics_json = Path(base_checkpoint_dir) / "dataset_statistics.json"
            with open(dataset_statistics_json, 'r') as f:
                norm_stats = json.load(f)

            vision_backbone = DinoSigLIPViTBackbone(
                image_resize_strategy="resize-naive",
                default_image_size=224,
                grid_size=16,
            )
            llm_backbone = LLaMa2LLMBackbone(
                llm_max_length=2048, hf_token="", inference_mode=False)
            tokenizer = llm_backbone.get_tokenizer()
            action_tokenizer = ActionTokenizer(tokenizer)

            vla = OpenFly(
                model_id="prism-dinosiglip-224px+7b",
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                arch_specifier="no-align+fused-gelu-mlp",
                norm_stats=norm_stats,
                action_tokenizer=action_tokenizer,
            )

            # 加载 base safetensors 权重
            index_file = Path(base_checkpoint_dir) / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
                state_dict = {}
                for shard_file in set(index['weight_map'].values()):
                    shard_path = Path(base_checkpoint_dir) / shard_file
                    if shard_path.exists():
                        state_dict.update(load_file(str(shard_path)))
                vla.load_state_dict(state_dict, strict=False)
                if self.verbose:
                    print(f"  ✓ Base VLA 权重加载完成")

            # ===== 加载 stage2 LoRA 权重 =====
            from peft import PeftModel
            vla.llm_backbone.llm = PeftModel.from_pretrained(
                vla.llm_backbone.llm,
                checkpoint_dir,
                is_trainable=True
            )
            if self.verbose:
                print(f"  ✓ Stage 2 LoRA 加载完成")

            # ===== 加载 projector =====
            projector_path = Path(checkpoint_dir) / "projector.pt"
            if projector_path.exists():
                vla.projector.load_state_dict(
                    torch.load(projector_path, map_location="cpu"))
                for param in vla.projector.parameters():
                    param.requires_grad = True
                if self.verbose:
                    print(f"  ✓ Projector 加载完成")

            # ===== 启用 gradient checkpointing =====
            if hasattr(vla.llm_backbone.llm, "gradient_checkpointing_enable"):
                vla.llm_backbone.llm.gradient_checkpointing_enable()
                if self.verbose:
                    print(f"  ✓ Gradient Checkpointing 已启用")

            return vla

        except Exception as e:
            raise RuntimeError(f"Stage 2 模型加载失败: {e}")

    def _load_auxiliary_heads(self, aux_heads_path: Path):
        """加载 stage2 的4个辅助头权重"""
        checkpoint = torch.load(aux_heads_path, map_location="cpu")

        self.keyword_head.load_state_dict(checkpoint['keyword_head'])
        self.direction_head.load_state_dict(checkpoint['direction_head'])
        self.cot_quality_head.load_state_dict(checkpoint['cot_quality_head'])
        self.action_validity_head.load_state_dict(checkpoint['action_validity_head'])

        if self.verbose:
            print(f"  ✓ 4个辅助头权重加载完成")
            print(f"  ✓ risk_head_1 和 risk_head_2 使用随机初始化")

    def _count_total_layers(self) -> int:
        """统计LLM backbone的总层数"""
        llm = self.base_vla.llm_backbone.llm
        if hasattr(llm, 'config') and hasattr(llm.config, 'num_hidden_layers'):
            return llm.config.num_hidden_layers
        layer_nums = set()
        import re
        for name, _ in llm.named_parameters():
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_nums.add(int(match.group(1)))
        return max(layer_nums) + 1 if layer_nums else 32

    def _setup_freezing_strategy(self, num_unfrozen_layers: int):
        """设置冻结策略（和stage2完全一致）"""
        total_layers = self._count_total_layers()
        unfrozen_start = total_layers - num_unfrozen_layers

        # 冻结 vision_backbone
        for param in self.base_vla.vision_backbone.parameters():
            param.requires_grad = False

        # projector 可训练
        if hasattr(self.base_vla, 'projector'):
            for param in self.base_vla.projector.parameters():
                param.requires_grad = True

        # LLM 先全部冻结
        llm = self.base_vla.llm_backbone.llm
        for param in llm.parameters():
            param.requires_grad = False

        # 解冻最后N层 + LoRA参数
        import re
        for name, param in llm.named_parameters():
            should_unfreeze = False
            for i in range(unfrozen_start, total_layers):
                if f'layers.{i}' in name:
                    should_unfreeze = True
                    break
            if 'lora' in name.lower():
                should_unfreeze = True
            if should_unfreeze:
                param.requires_grad = True

        if self.verbose:
            print(f"  ✓ 冻结策略设置完成（解冻最后{num_unfrozen_layers}层 + LoRA）")

    def _setup_auxiliary_heads(self):
        """初始化所有辅助heads（4个旧的 + 2个新的）"""

        # ===== Stage 2 原有的4个辅助头 =====
        self.keyword_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_keywords)
        )
        self.direction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 4)
        )
        self.cot_quality_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.action_validity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # ===== Stage 3 新增：Risk Heads =====

        # risk_head_1: 整体风险度（标量回归）
        # 训练信号：1 - normalized_chosen_score（全部1500样本）
        # 输出：0~1 的标量
        self.risk_head_1 = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # risk_head_2: 错误类型预测（4分类）
        # 类别：0=perception, 1=comprehension, 2=reasoning, 3=decision
        # 训练信号：仅有 aux_labels 的弱样本（约454个）
        # 输出：4维 logits（不加Softmax，CrossEntropy自带）
        self.risk_head_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 4)
        )

        if self.verbose:
            print(f"  ✓ 4个stage2辅助头初始化完成")
            print(f"  ✓ risk_head_1（整体风险度）初始化完成")
            print(f"  ✓ risk_head_2（错误类型预测）初始化完成")

    def _setup_hidden_state_hook(self):
        """设置hook以捕获LLM最后一层的hidden states（和stage2完全一致）"""
        self._captured_hidden_states = None
        self._hook_handle = None

        def hook_fn(module, input, output):
            try:
                hidden_states = None
                if isinstance(output, tuple) and len(output) > 0:
                    hidden_states = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    hidden_states = output.last_hidden_state
                elif isinstance(output, torch.Tensor):
                    hidden_states = output
                if hidden_states is not None and isinstance(hidden_states, torch.Tensor):
                    self._captured_hidden_states = hidden_states.detach()
            except Exception as e:
                if self.verbose:
                    print(f"  [Hook Warning] 捕获失败: {e}")

        try:
            llm_model = self.base_vla.llm_backbone.llm
            if hasattr(llm_model, 'base_model'):
                inner = llm_model.base_model
                if hasattr(inner, 'model'):
                    inner = inner.model
                if hasattr(inner, 'model'):
                    inner = inner.model
                if hasattr(inner, 'layers'):
                    last_layer = inner.layers[-1]
                    self._hook_handle = last_layer.register_forward_hook(hook_fn)
                    if self.verbose:
                        print(f"  ✓ Hook已注册 (PeftModel路径)")
                    return
            if hasattr(llm_model, 'model') and hasattr(llm_model.model, 'layers'):
                last_layer = llm_model.model.layers[-1]
                self._hook_handle = last_layer.register_forward_hook(hook_fn)
                if self.verbose:
                    print(f"  ✓ Hook已注册 (标准路径)")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Hook注册失败: {e}")

    def _remove_hidden_state_hook(self):
        if hasattr(self, '_hook_handle') and self._hook_handle is not None:
            self._hook_handle.remove()

    def __del__(self):
        self._remove_hidden_state_hook()

    def _extract_hidden_states_from_capture(self, batch_size: int) -> torch.Tensor:
        """从捕获的hidden states中提取句子级表示（和stage2完全一致）"""
        if self._captured_hidden_states is None:
            raise RuntimeError("未捕获到hidden states")

        hidden_states = self._captured_hidden_states

        if self._debug_count < 3:
            print(f"  [DEBUG] hidden_states.shape={hidden_states.shape}, hidden_dim={self.hidden_dim}")
            self._debug_count += 1

        if hidden_states.dim() == 3:
            b, s, h = hidden_states.shape
            if h == self.hidden_dim:
                pooled = hidden_states[:, -1, :]
            elif s == self.hidden_dim:
                pooled = hidden_states.transpose(1, 2)[:, -1, :]
            else:
                raise ValueError(f"无法处理的形状: {hidden_states.shape}")
        elif hidden_states.dim() == 2:
            d0, d1 = hidden_states.shape
            if d1 == self.hidden_dim:
                pooled = hidden_states if d0 == batch_size \
                    else hidden_states[-1:, :].expand(batch_size, -1)
            elif d0 == self.hidden_dim:
                pooled = hidden_states.T[-1:, :].expand(batch_size, -1)
            else:
                raise ValueError(f"无法处理的2D形状: {hidden_states.shape}")
        else:
            raise ValueError(f"不支持的维度数: {hidden_states.dim()}")

        if pooled.shape != (batch_size, self.hidden_dim):
            if pooled.shape[0] == 1 and batch_size > 1:
                pooled = pooled.expand(batch_size, -1)

        self._captured_hidden_states = None
        return pooled

    def forward(
        self,
        pixel_values: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        return_aux_outputs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            outputs: {
                'logits': [batch, seq_len, vocab_size],
                'aux_outputs': {
                    # stage2原有
                    'keywords': [batch, 34],
                    'direction': [batch, 4],
                    'cot_quality': [batch, 1],
                    'action_validity': [batch, 1],
                    # stage3新增
                    'overall_risk': [batch, 1],
                    'error_type_probs': [batch, 4]
                }
            }
        """
        batch_size = input_ids.shape[0]

        # 1. base_vla 前向传播
        base_outputs = self.base_vla(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **kwargs
        )

        outputs = {}
        if hasattr(base_outputs, 'logits'):
            outputs['logits'] = base_outputs.logits
        elif isinstance(base_outputs, tuple):
            outputs['logits'] = base_outputs[0]
        else:
            raise ValueError(f"无法提取logits，输出类型: {type(base_outputs)}")

        # 2. 辅助任务输出
        if return_aux_outputs:
            aux_outputs = None

            try:
                if self._captured_hidden_states is not None:
                    hidden_states = self._extract_hidden_states_from_capture(batch_size)

                    aux_outputs = {
                        # stage2 原有4个头
                        'keywords': self.keyword_head(hidden_states),
                        'direction': self.direction_head(hidden_states),
                        'cot_quality': self.cot_quality_head(hidden_states),
                        'action_validity': self.action_validity_head(hidden_states),
                        # stage3 新增2个 risk heads
                        'overall_risk': self.risk_head_1(hidden_states),       # [batch, 1]
                        'error_type_probs': self.risk_head_2(hidden_states)    # [batch, 4]
                    }

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  提取辅助输出失败: {e}")
                aux_outputs = None

            # Fallback：全零输出
            if aux_outputs is None:
                try:
                    device = outputs['logits'].device
                    dtype = outputs['logits'].dtype
                    aux_outputs = {
                        'keywords': torch.zeros(batch_size, self.num_keywords, device=device, dtype=dtype),
                        'direction': torch.zeros(batch_size, 4, device=device, dtype=dtype),
                        'cot_quality': torch.zeros(batch_size, 1, device=device, dtype=dtype),
                        'action_validity': torch.zeros(batch_size, 1, device=device, dtype=dtype),
                        'overall_risk': torch.zeros(batch_size, 1, device=device, dtype=dtype),
                        'error_type_probs': torch.zeros(batch_size, 4, device=device, dtype=dtype),
                        '_is_fallback': True
                    }
                except Exception as e2:
                    if self.verbose:
                        print(f"⚠️  Fallback也失败: {e2}")

            if aux_outputs is not None:
                outputs['aux_outputs'] = aux_outputs

        return outputs

    def _print_parameter_stats(self):
        """打印参数统计"""
        vision_params = sum(p.numel() for p in self.base_vla.vision_backbone.parameters())
        llm_params = sum(p.numel() for p in self.base_vla.llm_backbone.llm.parameters())
        llm_trainable = sum(p.numel() for p in self.base_vla.llm_backbone.llm.parameters()
                           if p.requires_grad)

        aux_params = sum(p.numel() for head in [
            self.keyword_head, self.direction_head,
            self.cot_quality_head, self.action_validity_head
        ] for p in head.parameters())

        risk_params = sum(p.numel() for head in [
            self.risk_head_1, self.risk_head_2
        ] for p in head.parameters())

        print(f"\n参数统计:")
        print(f"  Vision Backbone: {vision_params/1e6:.1f}M (冻结)")
        print(f"  LLM: {llm_params/1e6:.1f}M (可训练: {llm_trainable/1e6:.1f}M)")
        print(f"  Stage2 辅助Heads: {aux_params/1e6:.2f}M (可训练)")
        print(f"  Stage3 Risk Heads: {risk_params/1e6:.2f}M (可训练，随机初始化)")

    def save_checkpoint(self, save_path: str):
        """
        保存checkpoint

        保存内容：
        - LoRA 权重（HuggingFace格式）
        - projector.pt
        - auxiliary_heads.pt（6个heads，含2个risk heads）
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. 保存 LoRA 权重
        try:
            self.base_vla.llm_backbone.llm.save_pretrained(save_path)
        except Exception as e:
            print(f"⚠️ save_pretrained失败: {e}")
            lora_state = {k: v for k, v in
                         self.base_vla.llm_backbone.llm.state_dict().items()
                         if 'lora' in k.lower()}
            torch.save(lora_state, save_path / "adapter_model.bin")

        # 2. 保存 projector
        if hasattr(self.base_vla, 'projector'):
            torch.save(
                self.base_vla.projector.state_dict(),
                save_path / "projector.pt"
            )

        # 3. 保存所有辅助heads（6个，含risk heads）
        torch.save({
            # stage2 原有4个
            'keyword_head': self.keyword_head.state_dict(),
            'direction_head': self.direction_head.state_dict(),
            'cot_quality_head': self.cot_quality_head.state_dict(),
            'action_validity_head': self.action_validity_head.state_dict(),
            # stage3 新增2个
            'risk_head_1': self.risk_head_1.state_dict(),
            'risk_head_2': self.risk_head_2.state_dict(),
        }, save_path / "auxiliary_heads.pt")

        print(f"✅ Checkpoint 保存到: {save_path}")
        print(f"   - LoRA 权重")
        print(f"   - projector.pt")
        print(f"   - auxiliary_heads.pt（含6个heads）")
