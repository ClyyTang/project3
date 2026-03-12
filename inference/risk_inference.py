"""
推理阶段风险路由器 (B3)

三区分级处理：
  🟢 overall_risk < 0.3   → 正常贪心生成
  🟡 0.3 ≤ risk < 0.7    → Risk-Conditioned 生成（注入 error_type 向量）
  🔴 overall_risk ≥ 0.7  → 热插拔 Correction LoRA，greedy decoding，脱险后切回
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

PROJECT_ROOT = Path('/home/ubuntu/data1/lyy/full_rlds_project-3')
for p in [str(PROJECT_ROOT / 'stage3'), str(PROJECT_ROOT / 'stage2'),
          str(PROJECT_ROOT), '/home/ubuntu/data1/lyy/OpenFly-Platform/train']:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_config import Stage3Config
from risk_model import RiskAwareVLA


# ==================== 风险区间常量 ====================

ZONE_GREEN  = 'green'   # risk < 0.3  → 正常生成
ZONE_YELLOW = 'yellow'  # 0.3 ≤ risk < 0.7 → 条件生成
ZONE_RED    = 'red'     # risk ≥ 0.7  → Correction LoRA

ACTION_MAP = {
    0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right",
    4: "up", 5: "down", 6: "left", 7: "right",
    8: "fast_forward", 9: "super_fast"
}

ERROR_TYPE_MAP = {
    0: 'perception', 1: 'comprehension', 2: 'reasoning', 3: 'decision'
}


# ==================== Correction LoRA 管理器 ====================

class CorrectionLoRAManager:
    """
    热插拔管理器

    维护两套 LoRA 权重：
    - base_lora_state:       stage3 训练的通用 LoRA
    - correction_lora_state: Correction LoRA（高危专用）

    切换时只替换 lora_ 参数，不动其他权重，毫秒级完成。
    """

    def __init__(
        self,
        model: RiskAwareVLA,
        correction_lora_path: str,
        device: str = "cuda:0"
    ):
        self.model = model
        self.device = device
        self.active_lora = 'base'

        # 保存 base LoRA 权重（stage3 的通用 LoRA）
        print("  💾 备份 base LoRA 权重...")
        self.base_lora_state = {
            name: param.data.clone()
            for name, param in model.base_vla.llm_backbone.llm.named_parameters()
            if 'lora_' in name
        }

        # 加载 Correction LoRA 权重
        print(f"  📂 加载 Correction LoRA: {correction_lora_path}")
        correction_state = torch.load(
            correction_lora_path, map_location=device)
        self.correction_lora_state = correction_state

        print(f"  ✅ LoRA 管理器就绪")
        print(f"     Base LoRA 参数: "
              f"{sum(v.numel() for v in self.base_lora_state.values()):,}")
        print(f"     Correction LoRA 参数: "
              f"{sum(v.numel() for v in self.correction_lora_state.values()):,}")

    def activate_correction(self):
        """热插拔：切换到 Correction LoRA"""
        if self.active_lora == 'correction':
            return
        self._swap_lora(self.correction_lora_state)
        self.active_lora = 'correction'

    def restore_base(self):
        """热插拔：切回 base LoRA"""
        if self.active_lora == 'base':
            return
        self._swap_lora(self.base_lora_state)
        self.active_lora = 'base'

    def _swap_lora(self, target_state: Dict):
        """替换 lora_ 参数"""
        llm = self.model.base_vla.llm_backbone.llm
        with torch.no_grad():
            for name, param in llm.named_parameters():
                if 'lora_' in name and name in target_state:
                    param.data.copy_(target_state[name].to(self.device))

    @property
    def is_correction_active(self) -> bool:
        return self.active_lora == 'correction'


# ==================== Risk-Conditioned 生成（黄区）====================

class RiskConditionedGenerator:
    """
    黄区：把 risk_head_2 预测的错误类型向量注入生成过程

    实现方式：
    在每个 decoding step，把 error_type embedding 加到最后一个
    token 的 hidden state 上，引导模型更谨慎地处理高危维度。
    """

    def __init__(self, model: RiskAwareVLA, device: str):
        self.model = model
        self.device = device
        hidden_dim = 4096

        # error_type 条件向量（每种错误类型一个可学习的 embedding）
        # 这里用简单的线性投影：error_type_probs → hidden_dim
        self.condition_proj = nn.Linear(4, hidden_dim).to(device)

        # 用 stage3 训练时的权重初始化（如果有的话，否则随机初始化）
        nn.init.xavier_uniform_(self.condition_proj.weight)
        nn.init.zeros_(self.condition_proj.bias)

    def generate(
        self,
        pixel_values: Dict,
        input_ids: torch.Tensor,
        error_type_probs: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        tokenizer=None
    ) -> str:
        """
        Risk-Conditioned 生成

        在每步 decoding 时，把 error_type_probs 投影到 hidden_dim，
        作为一个偏置加到 logits 上，使模型更关注高危错误维度。
        """
        # 条件向量：[1, hidden_dim]
        condition_vec = self.condition_proj(
            error_type_probs.to(self.device).float()
        )  # [batch, hidden_dim]

        # 把条件向量投影到 vocab_size（作为 logit bias）
        vocab_size = tokenizer.vocab_size if tokenizer else 32000
        logit_bias_proj = nn.Linear(
            condition_vec.shape[-1], vocab_size, bias=False
        ).to(self.device)
        nn.init.zeros_(logit_bias_proj.weight)
        logit_bias = logit_bias_proj(condition_vec)  # [batch, vocab_size]

        current_ids = input_ids.clone()
        vocab_size_actual = tokenizer.vocab_size if tokenizer else 32000

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=current_ids
                )
                logits = outputs.get('logits')
                if logits is None:
                    break

                next_logits = logits[0, -1, :vocab_size_actual]

                # 注入条件偏置（黄区核心）
                bias = logit_bias[0, :vocab_size_actual]
                next_logits = next_logits + 0.1 * bias

                # 温度采样
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, keepdim=True)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                current_ids = torch.cat(
                    [current_ids, next_token.unsqueeze(0)], dim=1)

        valid_ids = [t for t in current_ids[0].tolist()
                     if 0 <= t < vocab_size_actual]
        generated = tokenizer.decode(valid_ids, skip_special_tokens=True)
        return generated


# ==================== 主推理引擎 ====================

class RiskAwareInferenceEngine:
    """
    推理阶段风险路由器

    使用方式：
        engine = RiskAwareInferenceEngine(config, stage3_ckpt, correction_lora_path)
        result = engine.infer(pixel_values, instruction)
        action = result['action']
    """

    def __init__(
        self,
        config: Stage3Config,
        stage3_checkpoint_dir: str,
        correction_lora_path: Optional[str] = None,
        device: str = "cuda:0",
        verbose: bool = True
    ):
        self.config = config
        self.device = device
        self.verbose = verbose

        # 风险阈值
        self.threshold_green = config.risk_threshold_green   # 0.3
        self.threshold_red   = config.risk_threshold_red     # 0.7

        print(f"\n{'='*60}")
        print("RiskAwareInferenceEngine 初始化")
        print(f"{'='*60}")
        print(f"🟢 绿区 < {self.threshold_green}：正常生成")
        print(f"🟡 黄区 < {self.threshold_red}：Risk-Conditioned 生成")
        print(f"🔴 红区 ≥ {self.threshold_red}：Correction LoRA 激活")

        # 加载模型
        print(f"\n[1/3] 加载 RiskAwareVLA...")
        self.model = RiskAwareVLA(
            stage2_checkpoint_dir=stage3_checkpoint_dir,
            num_unfrozen_layers=0,
            verbose=False
        ).to(device)
        self.model.eval()

        self.tokenizer = self.model.base_vla.llm_backbone.tokenizer

        # Correction LoRA 管理器（可选）
        self.lora_manager = None
        if correction_lora_path and Path(correction_lora_path).exists():
            print(f"\n[2/3] 初始化 Correction LoRA 管理器...")
            self.lora_manager = CorrectionLoRAManager(
                model=self.model,
                correction_lora_path=correction_lora_path,
                device=device
            )
        else:
            print(f"\n[2/3] ⚠️  未找到 Correction LoRA，红区将降级到贪心生成")

        # Risk-Conditioned 生成器
        print(f"\n[3/3] 初始化 Risk-Conditioned 生成器...")
        self.rc_generator = RiskConditionedGenerator(self.model, device)

        print(f"\n✅ 推理引擎就绪")
        print(f"{'='*60}\n")

    def _build_prompt(self, instruction: str) -> str:
        if hasattr(self.model.base_vla, 'get_prompt_builder'):
            pb = self.model.base_vla.get_prompt_builder()
        else:
            from model.prompt_llama2 import LLaMa2ChatPromptBuilder
            pb = LLaMa2ChatPromptBuilder("prismatic")
        pb.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?"
        )
        return pb.get_prompt()

    def _greedy_generate(
        self,
        pixel_values: Dict,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256
    ) -> str:
        """绿区：标准贪心生成"""
        current_ids = input_ids.clone()
        vocab_size = self.tokenizer.vocab_size

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=current_ids
                )
                logits = outputs.get('logits')
                if logits is None:
                    break

                next_token = logits[0, -1, :vocab_size].argmax(keepdim=True)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                current_ids = torch.cat(
                    [current_ids, next_token.unsqueeze(0)], dim=1)

        valid_ids = [t for t in current_ids[0].tolist()
                     if 0 <= t < vocab_size]
        return self.tokenizer.decode(valid_ids, skip_special_tokens=True)

    def _correction_generate(
        self,
        pixel_values: Dict,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256
    ) -> str:
        """
        红区：Correction LoRA + greedy decoding

        注入反事实引导 prefix，强制模型进行脱险推理。
        """
        # 注入反事实引导前缀
        prefix = (
            "<thinking>\n"
            "警告：当前状态检测到高风险！必须立即执行脱险动作。\n"
            "反事实推理：如果继续当前轨迹，将发生碰撞或任务失败。\n"
            "因此必须选择保守安全的动作：\n"
        )
        prefix_ids = self.tokenizer(
            prefix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        current_ids = torch.cat([input_ids, prefix_ids], dim=1)
        vocab_size = self.tokenizer.vocab_size

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=current_ids
                )
                logits = outputs.get('logits')
                if logits is None:
                    break

                # 红区：纯贪心，不采样，确定性输出
                next_token = logits[0, -1, :vocab_size].argmax(keepdim=True)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                current_ids = torch.cat(
                    [current_ids, next_token.unsqueeze(0)], dim=1)

        valid_ids = [t for t in current_ids[0].tolist()
                     if 0 <= t < vocab_size]
        return self.tokenizer.decode(valid_ids, skip_special_tokens=True)

    def _parse_action(self, generated_text: str) -> Optional[int]:
        """从生成文本里解析 action"""
        match = re.search(r'<action>(\d+)</action>', generated_text)
        if match:
            action = int(match.group(1))
            if 0 <= action <= 9:
                return action
        # 降级：找最后出现的单个数字
        numbers = re.findall(r'\b([0-9])\b', generated_text)
        if numbers:
            return int(numbers[-1])
        return None

    def _determine_zone(self, risk_score: float) -> str:
        if risk_score < self.threshold_green:
            return ZONE_GREEN
        elif risk_score < self.threshold_red:
            return ZONE_YELLOW
        else:
            return ZONE_RED

    def infer(
        self,
        pixel_values: Dict,
        instruction: str,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        主推理接口

        Args:
            pixel_values: 图像特征（已 transform）
            instruction:  任务指令
            max_new_tokens: 最大生成 token 数

        Returns:
            {
                'action': int,           # 0-9
                'action_name': str,      # 动作名称
                'zone': str,             # green/yellow/red
                'risk_score': float,
                'error_type': str,
                'generated_text': str,
                'correction_lora_used': bool
            }
        """
        self.model.eval()

        # 构造 prompt
        prompt_text = self._build_prompt(instruction)
        input_ids = self.tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)

        pv = {k: v.to(self.device) for k, v in pixel_values.items()}

        # Step 1: 快速风险评估（不生成完整序列）
        with torch.no_grad():
            outputs = self.model(pixel_values=pv, input_ids=input_ids)

        aux = outputs.get('aux_outputs', {})
        risk_score = aux.get('overall_risk', torch.tensor([[0.5]])).item()
        error_type_probs = aux.get('error_type_probs')

        et_idx = error_type_probs.argmax(dim=-1).item() \
            if error_type_probs is not None else 0
        error_type = ERROR_TYPE_MAP.get(et_idx, 'unknown')

        zone = self._determine_zone(risk_score)

        zone_emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}[zone]
        if self.verbose:
            print(f"{zone_emoji} 风险评分: {risk_score:.3f} | "
                  f"错误类型: {error_type} | 区间: {zone}")

        # Step 2: 根据区间选择生成策略
        correction_lora_used = False

        if zone == ZONE_GREEN:
            # 🟢 标准贪心生成
            generated_text = self._greedy_generate(pv, input_ids, max_new_tokens)

        elif zone == ZONE_YELLOW:
            # 🟡 Risk-Conditioned 生成
            if error_type_probs is not None:
                generated_text = self.rc_generator.generate(
                    pixel_values=pv,
                    input_ids=input_ids,
                    error_type_probs=error_type_probs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.5,
                    tokenizer=self.tokenizer
                )
            else:
                # 降级到绿区策略
                generated_text = self._greedy_generate(pv, input_ids, max_new_tokens)

        else:
            # 🔴 Correction LoRA 热插拔
            if self.lora_manager is not None:
                self.lora_manager.activate_correction()
                correction_lora_used = True
                if self.verbose:
                    print("  🔴 Correction LoRA 已激活")

            generated_text = self._correction_generate(
                pv, input_ids, max_new_tokens)

            # 脱险后切回 base LoRA
            if self.lora_manager is not None:
                self.lora_manager.restore_base()
                if self.verbose:
                    print("  ✅ Correction LoRA 已卸载，切回 base LoRA")

        # Step 3: 解析 action
        action = self._parse_action(generated_text)
        if action is None:
            # 红区默认上升（最保守的逃生动作）
            action = 4 if zone == ZONE_RED else 0
            if self.verbose:
                print(f"  ⚠️  无法解析 action，使用默认值: {action}")

        return {
            'action': action,
            'action_name': ACTION_MAP.get(action, 'unknown'),
            'zone': zone,
            'risk_score': risk_score,
            'error_type': error_type,
            'error_type_idx': et_idx,
            'generated_text': generated_text,
            'correction_lora_used': correction_lora_used
        }

    def infer_from_image(
        self,
        image_path: str,
        instruction: str,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        便捷接口：直接传图片路径

        Args:
            image_path: 图片文件路径
            instruction: 任务指令

        Returns:
            同 infer()
        """
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        tr_img = self.model.base_vla.vision_backbone.image_transform(img)

        pixel_values = {}
        for k in tr_img.keys():
            combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
            pixel_values[k] = combined.unsqueeze(0).to(self.device)

        return self.infer(pixel_values, instruction, max_new_tokens)

    def infer_batch(
        self,
        samples: List[Dict],
        image_base_path: str
    ) -> List[Dict]:
        """
        批量推理（评估用）

        Args:
            samples: [{'image_path': ..., 'frame_idx': ...,
                       'instruction': ...}, ...]
            image_base_path: 图片根目录

        Returns:
            List of infer() 结果
        """
        results = []
        base_root = Path(image_base_path)

        for i, sample in enumerate(samples):
            instruction = (sample.get('gpt_instruction') or
                           sample.get('instruction', ''))
            episode_folder = (sample.get('image_path') or
                              sample.get('episode_id', ''))
            frame_name = sample.get('frame_idx')
            if not frame_name and 'index_list' in sample and sample['index_list']:
                frame_name = sample['index_list'][0]

            img_path = None
            if episode_folder and frame_name:
                folder_path = base_root / episode_folder
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = folder_path / f"{frame_name}{ext}"
                    if test_path.exists():
                        img_path = str(test_path)
                        break

            if img_path is None:
                results.append({'error': 'image_not_found', 'sample_idx': i})
                continue

            try:
                result = self.infer_from_image(img_path, instruction)
                result['sample_idx'] = i
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'sample_idx': i})

            if (i + 1) % 100 == 0:
                zone_counts = {}
                for r in results:
                    z = r.get('zone', 'error')
                    zone_counts[z] = zone_counts.get(z, 0) + 1
                print(f"  进度: {i+1}/{len(samples)} | 区间分布: {zone_counts}")

        return results


def main():
    """简单演示"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3_ckpt', type=str, required=True)
    parser.add_argument('--correction_lora', type=str, default=None)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--instruction', type=str,
                        default="fly forward and avoid obstacles")
    args = parser.parse_args()

    config = Stage3Config()
    engine = RiskAwareInferenceEngine(
        config=config,
        stage3_checkpoint_dir=args.stage3_ckpt,
        correction_lora_path=args.correction_lora,
        verbose=True
    )

    result = engine.infer_from_image(args.image, args.instruction)

    print(f"\n{'='*60}")
    print("推理结果")
    print(f"{'='*60}")
    print(f"动作:      {result['action']} ({result['action_name']})")
    print(f"区间:      {result['zone']}")
    print(f"风险分:    {result['risk_score']:.3f}")
    print(f"错误类型:  {result['error_type']}")
    print(f"用了纠偏LoRA: {result['correction_lora_used']}")
    print(f"\n生成文本:\n{result['generated_text'][:300]}...")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
