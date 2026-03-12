"""
Multi-task VLA模型 (MultiTaskVLA)

功能：在Stage 1 VLA基础上添加辅助任务heads，支持Multi-task Learning
架构：
- 基础VLA模型（OpenVLA）- 部分冻结
- 4个辅助任务heads（可训练）

使用场景：Stage 2 GSPO训练
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# 添加OpenFly-Platform到路径
OPENFLY_PATH = '/home/ubuntu/data1/lyy/OpenFly-Platform'
if OPENFLY_PATH not in sys.path:
    sys.path.insert(0, os.path.join(OPENFLY_PATH, 'train'))

# 导入OpenVLA相关类（从test_inference.py学习）
try:
    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoImageProcessor, AutoProcessor
    from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    
    # 注册自定义类
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)
    
    OPENVLA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入OpenVLA模块: {e}")
    print(f"   请确保OpenFly-Platform在正确路径: {OPENFLY_PATH}")
    OPENVLA_AVAILABLE = False
    OpenVLAForActionPrediction = None


class MultiTaskVLA(nn.Module):
    """
    Multi-task VLA模型
    
    架构：
    - base_vla: Stage 1训练的VLA模型（部分冻结）
    - keyword_head: 关键词识别（34维输出）
    - direction_head: 方向分类（4维输出）
    - cot_quality_head: CoT质量评分（1维输出）
    - action_validity_head: 动作合理性（1维输出）
    """
    
    def __init__(
        self,
        stage1_checkpoint_dir: str,
        num_unfrozen_layers: int = 4,
        num_keywords: int = 34,
        hidden_dim: int = 4096,
        dropout: float = 0.1,
        verbose: bool = True
    ):
        """
        初始化Multi-task VLA模型
        
        Args:
            stage1_checkpoint_dir: Stage 1 checkpoint目录路径
            num_unfrozen_layers: 解冻的层数（从后往前）
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
            print("初始化Multi-task VLA模型")
            print(f"{'='*60}")
        
        # 1. 检查OpenVLA是否可用
        if not OPENVLA_AVAILABLE:
            raise RuntimeError("OpenVLA模块不可用，请检查OpenFly-Platform路径")
        
        # 2. 加载Stage 1模型
        if self.verbose:
            print(f"加载Stage 1模型: {stage1_checkpoint_dir}")
        
        self.base_vla = self._load_stage1_model(stage1_checkpoint_dir)
        
        # 3. 设置冻结策略
        if self.verbose:
            print(f"\n设置冻结策略: 解冻最后{num_unfrozen_layers}层")
        
        self._setup_freezing_strategy(num_unfrozen_layers)
        
        # 4. 初始化辅助任务heads
        if self.verbose:
            print(f"\n初始化辅助任务heads:")
            print(f"  - Keyword head: {hidden_dim} → 512 → {num_keywords}")
            print(f"  - Direction head: {hidden_dim} → 256 → 4")
            print(f"  - CoT quality head: {hidden_dim} → 256 → 1")
            print(f"  - Action validity head: {hidden_dim} → 256 → 1")
        
        self._setup_auxiliary_heads()
        
        # 5. 设置hidden state hook（用于forward时捕获hidden states）
        if self.verbose:
            print(f"\n设置hidden state hook:")
        
        self._setup_hidden_state_hook()
        
        # 6. 统计参数
        if self.verbose:
            self._print_parameter_stats()
            print(f"\n{'='*60}")
            print("✅ Multi-task VLA模型初始化完成")
            print(f"{'='*60}\n")
        
        # 调试计数器
        self._debug_count = 0
    
    def _load_stage1_model(self, checkpoint_dir: str) -> nn.Module:
        """
        加载Stage 1训练的VLA模型
        """
        # 检查路径是否存在
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")
        
        try:
            # ===== Step 0: Monkey patch =====
            if self.verbose:
                print(f"  Step 0: 应用monkey patch...")
            
            import model.llm_backbone as llm_backbone_module
            from transformers import LlamaForCausalLM, AutoConfig
            
            def patched_init(self, llm_max_length, hf_token=None, use_flash_attention_2=False, inference_mode=False):
                super(llm_backbone_module.LLaMa2LLMBackbone, self).__init__()
                self.llm_max_length = llm_max_length
                self.inference_mode = inference_mode
                
                hf_hub_path = "/home/ubuntu/data1/zx/OpenFly-Platform/train/meta-llama/llama2-7b-hf"
                
                if not inference_mode:
                    self.llm = LlamaForCausalLM.from_pretrained(
                        hf_hub_path, 
                        local_files_only=True, 
                        do_sample=False, 
                        temperature=1.0, 
                        top_p=1.0
                    )
                else:
                    llm_config = AutoConfig.from_pretrained(hf_hub_path, local_files_only=True)
                    self.llm = LlamaForCausalLM._from_config(llm_config)
                
                self.tokenizer = llm_backbone_module.AutoTokenizer.from_pretrained(
                    hf_hub_path, 
                    local_files_only=True, 
                    model_max_length=llm_max_length, 
                    padding_side="right", 
                    use_fast=False
                )
                self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
                self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
            
            # 应用patch
            llm_backbone_module.LLaMa2LLMBackbone.__init__ = patched_init
            
            if self.verbose:
                print(f"  ✓ Monkey patch已应用")
            
            # ===== Step 1: 初始化overwatch =====
            if self.verbose:
                print(f"  Step 1: 初始化overwatch...")
            
            from model.overwatch import initialize_overwatch
            overwatch = initialize_overwatch("multitask-vla")
            
            if self.verbose:
                print(f"  ✓ Overwatch初始化完成")
            
            # ===== Step 2: 加载HuggingFace格式的OpenFly模型 =====
            if self.verbose:
                print(f"  Step 2: 加载原始base VLA模型...")
                print(f"  从HuggingFace格式目录加载")
            
            # 模型路径
            base_checkpoint_dir = "/home/ubuntu/data1/lyy/openfly-agent-7b"
            
            # 检查必要文件
            config_json = Path(base_checkpoint_dir) / "config.json"
            if not config_json.exists():
                raise FileNotFoundError(f"Missing config.json: {config_json}")
            
            from model.load_model import OpenFly
            from model.vision_backbone import DinoSigLIPViTBackbone
            from model.action_tokenizer import ActionTokenizer
            from model.llm_backbone import LLaMa2LLMBackbone
            from safetensors.torch import load_file
            
            # 加载统计信息
            dataset_statistics_json = Path(base_checkpoint_dir) / "dataset_statistics.json"
            with open(dataset_statistics_json, 'r') as f:
                norm_stats = json.load(f)
            
            # 创建组件
            vision_backbone = DinoSigLIPViTBackbone(
                image_resize_strategy="resize-naive",
                default_image_size=224,
                grid_size=16,
            )
            
            llm_backbone = LLaMa2LLMBackbone(
                llm_max_length=2048,
                hf_token="",
                inference_mode=False,
            )
            
            tokenizer = llm_backbone.get_tokenizer()
            action_tokenizer = ActionTokenizer(tokenizer)
            
            # 实例化OpenFly
            vla = OpenFly(
                model_id="prism-dinosiglip-224px+7b",
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                arch_specifier="no-align+fused-gelu-mlp",
                norm_stats=norm_stats,
                action_tokenizer=action_tokenizer,
            )
            
            # 手动加载safetensors权重
            if self.verbose:
                print(f"  加载safetensors权重...")
            
            # 读取index找到所有分片
            index_file = Path(base_checkpoint_dir) / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                # 加载所有分片
                state_dict = {}
                weight_map = index.get('weight_map', {})
                shard_files = set(weight_map.values())
                
                for shard_file in shard_files:
                    shard_path = Path(base_checkpoint_dir) / shard_file
                    if shard_path.exists():
                        shard_dict = load_file(str(shard_path))
                        state_dict.update(shard_dict)
                
                if self.verbose:
                    print(f"  ✓ 加载了{len(shard_files)}个分片")
                
                # 加载到模型
                vla.load_state_dict(state_dict, strict=False)
                
                if self.verbose:
                    print(f"  ✓ 权重加载完成")
            else:
                # 没有index，尝试直接加载safetensors
                safetensors_file = Path(base_checkpoint_dir) / "model.safetensors"
                if safetensors_file.exists():
                    state_dict = load_file(str(safetensors_file))
                    vla.load_state_dict(state_dict, strict=False)
                else:
                    raise FileNotFoundError("找不到safetensors文件")
            
            if self.verbose:
                print(f"  ✓ Base VLA加载成功")
            
            # ===== Step 3: 加载Stage 1 LoRA权重 =====
            if self.verbose:
                print(f"  Step 3: 加载Stage 1 LoRA权重...")
                print(f"         从: {checkpoint_dir}")
            
            from peft import PeftModel
            
            vla.llm_backbone.llm = PeftModel.from_pretrained(
                vla.llm_backbone.llm,
                checkpoint_dir,
                is_trainable=True
            )
            
            if self.verbose:
                print(f"  ✓ Stage 1 LoRA加载成功")
            
            # ===== Step 4: 加载projector（如果存在） =====
            projector_path = Path(checkpoint_dir) / "projector.pt"
            if projector_path.exists():
                if self.verbose:
                    print(f"  Step 4: 加载projector...")
                
                vla.projector.load_state_dict(
                    torch.load(projector_path, map_location="cpu")
                )
                
                # 设置projector可训练
                for param in vla.projector.parameters():
                    param.requires_grad = True
                
                if self.verbose:
                    print(f"  ✓ Projector加载成功")
            
            # ===== Step 5: 启用gradient checkpointing =====
            if hasattr(vla.llm_backbone.llm, "gradient_checkpointing_enable"):
                vla.llm_backbone.llm.gradient_checkpointing_enable()
                if self.verbose:
                    print(f"  ✓ Gradient Checkpointing已启用")
            
            if self.verbose:
                print(f"  ✓ 模型加载完成")
                print(f"  ✓ 模型类型: {type(vla).__name__}")
            
            return vla
            
        except ImportError as e:
            raise ImportError(
                f"无法导入必要模块: {e}\n"
                f"请确保OpenFly-Platform路径正确，并且已安装peft库"
            )
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _count_total_layers(self) -> int:
        """统计LLM backbone的总层数"""
        llm = self.base_vla.llm_backbone.llm
        
        # 尝试从config获取
        if hasattr(llm, 'config') and hasattr(llm.config, 'num_hidden_layers'):
            total_layers = llm.config.num_hidden_layers
            if self.verbose:
                print(f"  ✓ 从config检测到{total_layers}层")
            return total_layers
        
        # 尝试从参数名推断
        layer_nums = set()
        for name, _ in llm.named_parameters():
            import re
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_nums.add(int(match.group(1)))
        
        if len(layer_nums) > 0:
            total_layers = max(layer_nums) + 1
            if self.verbose:
                print(f"  ✓ 从参数名检测到{total_layers}层")
            return total_layers
        
        # 默认值（Llama-2-7B是32层）
        if self.verbose:
            print(f"  ⚠️  未找到层数，使用默认值32")
        return 32
    
    def _setup_freezing_strategy(self, num_unfrozen_layers: int):
        """设置冻结策略"""
        # 统计总层数
        total_layers = self._count_total_layers()
        unfrozen_start = total_layers - num_unfrozen_layers
        
        if self.verbose:
            print(f"\n  冻结策略:")
            print(f"  - Vision Backbone: 全部冻结")
            print(f"  - Projector: 全部可训练")
            print(f"  - LLM层{unfrozen_start}-{total_layers-1}: 解冻")
            print(f"  - LoRA参数: 可训练\n")
        
        # 1. 冻结vision_backbone
        for param in self.base_vla.vision_backbone.parameters():
            param.requires_grad = False
        
        # 2. projector设为可训练（如果存在）
        if hasattr(self.base_vla, 'projector'):
            for param in self.base_vla.projector.parameters():
                param.requires_grad = True
        
        # 3. LLM部分：先全部冻结
        llm = self.base_vla.llm_backbone.llm
        for param in llm.parameters():
            param.requires_grad = False
        
        # 4. 解冻最后N层 + LoRA参数
        unfrozen_count = 0
        lora_count = 0
        
        for name, param in llm.named_parameters():
            should_unfreeze = False
            
            # 检查是否在解冻的层范围内
            import re
            for i in range(unfrozen_start, total_layers):
                if f'layers.{i}' in name:
                    should_unfreeze = True
                    break
            
            # 检查是否是LoRA参数
            if 'lora' in name.lower():
                should_unfreeze = True
                lora_count += 1
            
            # 解冻
            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += 1
                
                if self.verbose and unfrozen_count <= 5:
                    print(f"  ✓ 解冻: {name[:80]}...")
        
        if self.verbose:
            if unfrozen_count > 5:
                print(f"  ... (省略{unfrozen_count - 5}个参数)")
            print(f"\n  LLM解冻参数: {unfrozen_count}")
            if lora_count > 0:
                print(f"  其中LoRA参数: {lora_count}")
    
    def _setup_auxiliary_heads(self):
        """初始化辅助任务heads"""
        # 1. Keyword head (多标签分类)
        self.keyword_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_keywords)
        )
        
        # 2. Direction head (单标签分类)
        self.direction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 4)
        )
        
        # 3. CoT quality head (回归)
        self.cot_quality_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 4. Action validity head (回归)
        self.action_validity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _print_parameter_stats(self):
        """打印参数统计信息"""
        vision_params = sum(p.numel() for p in self.base_vla.vision_backbone.parameters())
        vision_trainable = sum(p.numel() for p in self.base_vla.vision_backbone.parameters() if p.requires_grad)
        
        projector_params = 0
        projector_trainable = 0
        if hasattr(self.base_vla, 'projector'):
            projector_params = sum(p.numel() for p in self.base_vla.projector.parameters())
            projector_trainable = sum(p.numel() for p in self.base_vla.projector.parameters() if p.requires_grad)
        
        llm_params = sum(p.numel() for p in self.base_vla.llm_backbone.llm.parameters())
        llm_trainable = sum(p.numel() for p in self.base_vla.llm_backbone.llm.parameters() if p.requires_grad)
        
        aux_params = 0
        for head in [self.keyword_head, self.direction_head, 
                     self.cot_quality_head, self.action_validity_head]:
            for param in head.parameters():
                aux_params += param.numel()
        
        total_params = vision_params + projector_params + llm_params + aux_params
        trainable_params = vision_trainable + projector_trainable + llm_trainable + aux_params
        
        print(f"\n参数统计:")
        print(f"  Vision Backbone: {vision_params/1e6:.1f}M (可训练: {vision_trainable/1e6:.1f}M)")
        print(f"  Projector: {projector_params/1e6:.1f}M (可训练: {projector_trainable/1e6:.1f}M)")
        print(f"  LLM Backbone: {llm_params/1e6:.1f}M (可训练: {llm_trainable/1e6:.1f}M)")
        print(f"  辅助Heads: {aux_params/1e6:.2f}M (全部可训练)")
        print(f"  ─────────────────────────")
        print(f"  总参数: {total_params/1e6:.1f}M")
        print(f"  可训练参数: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")
    
    def _setup_hidden_state_hook(self):
        """设置hook以捕获LLM最后一层的hidden states"""
        self._captured_hidden_states = None
        self._hook_handle = None
        
        def hook_fn(module, input, output):
            """Hook函数，捕获decoder layer的输出"""
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
        
        # 在LLM的最后一层注册hook
        try:
            llm_model = self.base_vla.llm_backbone.llm
            
            # 如果是PeftModel，需要正确访问内部模型
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
            
            # 非PeftModel情况
            if hasattr(llm_model, 'model') and hasattr(llm_model.model, 'layers'):
                last_layer = llm_model.model.layers[-1]
                self._hook_handle = last_layer.register_forward_hook(hook_fn)
                if self.verbose:
                    print(f"  ✓ Hook已注册 (标准路径)")
                return
            
            if self.verbose:
                print(f"  ⚠️ 无法找到LLM layers，将使用fallback方案")
                
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Hook注册失败: {e}，将使用fallback方案")
    
    def _remove_hidden_state_hook(self):
        """移除hook"""
        if hasattr(self, '_hook_handle') and self._hook_handle is not None:
            self._hook_handle.remove()
    
    def __del__(self):
        """析构时清理hook"""
        self._remove_hidden_state_hook()
    
    def _extract_hidden_states_from_capture(
        self,
        batch_size: int
    ) -> torch.Tensor:
        """
        从捕获的hidden states中提取句子级表示
        
        Args:
            batch_size: batch大小
            
        Returns:
            pooled_hidden: (batch_size, hidden_dim)
        """
        if self._captured_hidden_states is None:
            raise RuntimeError("未捕获到hidden states")
        
        hidden_states = self._captured_hidden_states
        
        # 打印调试信息（前几次）
        if self._debug_count < 3:
            print(f"  [DEBUG] hidden_states.shape = {hidden_states.shape}, hidden_dim = {self.hidden_dim}")
            self._debug_count += 1
        
        # 处理不同形状情况
        if hidden_states.dim() == 3:
            # 期望: [batch, seq_len, hidden_dim]
            b, s, h = hidden_states.shape
            
            if h == self.hidden_dim:
                # 正常情况: [batch, seq_len, hidden_dim]
                # 使用最后一个token的hidden state
                pooled = hidden_states[:, -1, :]  # [batch, hidden_dim]
            elif s == self.hidden_dim:
                # 维度被交换了: [batch, hidden_dim, seq_len]
                hidden_states = hidden_states.transpose(1, 2)  # -> [batch, seq_len, hidden_dim]
                pooled = hidden_states[:, -1, :]  # [batch, hidden_dim]
            else:
                raise ValueError(f"无法处理的形状: {hidden_states.shape}, hidden_dim={self.hidden_dim}")
        
        elif hidden_states.dim() == 2:
            d0, d1 = hidden_states.shape
            
            if d1 == self.hidden_dim:
                if d0 == batch_size:
                    pooled = hidden_states
                else:
                    pooled = hidden_states[-1:, :].expand(batch_size, -1)
            elif d0 == self.hidden_dim:
                pooled = hidden_states.T[-1:, :].expand(batch_size, -1)
            else:
                raise ValueError(f"无法处理的2D形状: {hidden_states.shape}")
        
        else:
            raise ValueError(f"不支持的维度数: {hidden_states.dim()}")
        
        # 最终验证
        if pooled.shape != (batch_size, self.hidden_dim):
            if pooled.shape[0] == 1 and batch_size > 1:
                pooled = pooled.expand(batch_size, -1)
            
            if pooled.shape != (batch_size, self.hidden_dim):
                raise ValueError(f"形状修复失败: {pooled.shape} != ({batch_size}, {self.hidden_dim})")
        
        # 清理
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
        
        Args:
            pixel_values: 图像输入 (Dict，键取决于vision backbone)
            input_ids: token ids [batch, seq_len]
            return_aux_outputs: 是否返回辅助任务输出
            **kwargs: 其他参数传递给base_vla
            
        Returns:
            outputs: {
                'logits': [batch, seq_len, vocab_size],
                'aux_outputs': {...} (如果return_aux_outputs=True)
            }
        """
        batch_size = input_ids.shape[0]
        
        # 1. 调用base_vla获取主任务输出
        base_outputs = self.base_vla(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **kwargs
        )
        
        # 2. 构建输出字典
        outputs = {}
        
        # 提取logits
        if hasattr(base_outputs, 'logits'):
            outputs['logits'] = base_outputs.logits
        elif isinstance(base_outputs, tuple):
            outputs['logits'] = base_outputs[0]
        else:
            raise ValueError(f"无法从base_vla输出中提取logits，输出类型: {type(base_outputs)}")
        
        # 3. 如果需要辅助任务输出
        if return_aux_outputs:
            aux_outputs = None
            
            try:
                # 方案1: 从hook捕获的hidden states中提取
                if self._captured_hidden_states is not None:
                    hidden_states = self._extract_hidden_states_from_capture(batch_size)
                    
                    # 验证形状
                    assert hidden_states.shape == (batch_size, self.hidden_dim), \
                        f"hidden_states形状错误: {hidden_states.shape}"
                    
                    # 通过辅助heads
                    aux_outputs = {
                        'keywords': self.keyword_head(hidden_states),
                        'direction': self.direction_head(hidden_states),
                        'cot_quality': self.cot_quality_head(hidden_states),
                        'action_validity': self.action_validity_head(hidden_states)
                    }
            
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  提取辅助任务输出失败: {e}")
                aux_outputs = None
            
            # 方案2: Fallback - 创建零输出作为placeholder
            if aux_outputs is None:
                try:
                    device = outputs['logits'].device
                    dtype = outputs['logits'].dtype
                    
                    aux_outputs = {
                        'keywords': torch.zeros(batch_size, self.num_keywords, device=device, dtype=dtype),
                        'direction': torch.zeros(batch_size, 4, device=device, dtype=dtype),
                        'cot_quality': torch.zeros(batch_size, 1, device=device, dtype=dtype),
                        'action_validity': torch.zeros(batch_size, 1, device=device, dtype=dtype)
                    }
                    
                    # 标记为fallback
                    aux_outputs['_is_fallback'] = True
                    
                    if self._debug_count < 5:
                        print(f"  使用零值fallback（辅助loss将被跳过）")
                        
                except Exception as e2:
                    if self.verbose:
                        print(f"⚠️  Fallback也失败: {e2}")
            
            if aux_outputs is not None:
                outputs['aux_outputs'] = aux_outputs
        
        return outputs
    
    def save_for_eval(self, save_path: str):
        """保存为eval兼容格式"""
        print(f"\n保存模型为eval格式: {save_path}")
        
        if not save_path.endswith('.pt'):
            os.makedirs(save_path, exist_ok=True)
            self.base_vla.save_pretrained(save_path)
            print(f"✓ 模型已保存至: {save_path}")
        else:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(self.base_vla.state_dict(), save_path)
            print(f"✓ 模型权重已保存至: {save_path}")
    
    def save_full_model(self, save_path: str):
        """保存完整模型（包括辅助heads）"""
        print(f"\n保存完整Multi-task模型: {save_path}")
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'base_vla_state_dict': self.base_vla.state_dict(),
            'keyword_head_state_dict': self.keyword_head.state_dict(),
            'direction_head_state_dict': self.direction_head.state_dict(),
            'cot_quality_head_state_dict': self.cot_quality_head.state_dict(),
            'action_validity_head_state_dict': self.action_validity_head.state_dict(),
            'config': {
                'num_unfrozen_layers': self.num_unfrozen_layers,
                'num_keywords': self.num_keywords,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout
            }
        }
        
        torch.save(checkpoint, save_path)
        
        print(f"✓ 完整模型已保存至: {save_path}")
        print(f"  包含base_vla + 4个辅助heads")


# 测试代码
if __name__ == '__main__':
    print("【测试Multi-task VLA模型初始化】\n")
    
    STAGE1_CHECKPOINT = '/home/ubuntu/data1/lyy/full_rlds_project/3_training/checkpoints/stage1_sft'
    
    if not os.path.exists(STAGE1_CHECKPOINT):
        print(f"⚠️  Checkpoint不存在: {STAGE1_CHECKPOINT}")
        print("   请修改路径后重试")
    else:
        try:
            model = MultiTaskVLA(
                stage1_checkpoint_dir=STAGE1_CHECKPOINT,
                num_unfrozen_layers=4,
                num_keywords=34,
                hidden_dim=4096,
                dropout=0.1,
                verbose=True
            )
            
            print("\n✅ 模型初始化成功！")
            
        except Exception as e:
            print(f"\n❌ 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()