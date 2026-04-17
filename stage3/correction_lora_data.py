"""
Correction LoRA 训练数据构造 (B1)

流程：
1. 加载 stage3 训完的 RiskAwareVLA
2. 遍历1500个样本，risk_head_1 打分
3. 找出 overall_risk >= 0.7 的高危帧
4. 用 CounterfactualFinder 找相似成功样本作为参考
5. 调 Qwen 生成反事实脱险 CoT
6. 保存为 correction_lora_train.json
"""

import os
import sys
import json
import torch
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = Path('/home/ubuntu/data1/lyy/full_rlds_project-3')
for p in [str(PROJECT_ROOT / 'stage3'), str(PROJECT_ROOT / 'stage2'),
          str(PROJECT_ROOT), '/home/ubuntu/data1/lyy/OpenFly-Platform/train']:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_config import Stage3Config
from risk_model import RiskAwareVLA
from counterfactual_finder import CounterfactualFinder


# ==================== Qwen 调用 ====================

def call_qwen(
    prompt: str,
    image_path: Optional[str] = None,
    api_base: str = "http://localhost:8000",
    model_name: str = "Qwen3-VL-32B-Instruct",
    max_tokens: int = 512,
    temperature: float = 0.3
) -> str:
    """调用 Qwen 生成反事实脱险 CoT"""
    messages = []

    if image_path and Path(image_path).exists():
        import base64
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((512, 512))
        import io
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{api_base}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"  ⚠️  Qwen 调用失败: {e}")
        return ""


def build_correction_prompt(
    instruction: str,
    risk_score: float,
    error_type: str,
    counterfactuals: List[Dict],
    current_action: Optional[int] = None
) -> str:
    """
    构造反事实脱险 CoT 的 prompt

    核心思路：
    - 告知模型当前状态危险（risk_score, error_type）
    - 提供相似成功案例作为参考（counterfactuals）
    - 要求生成带反事实推理的脱险 CoT
    """
    action_map = {
        0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right",
        4: "up", 5: "down", 6: "left", 7: "right",
        8: "fast_forward", 9: "super_fast"
    }

    error_desc = {
        'perception': '感知失效（视觉信息误判）',
        'comprehension': '指令理解偏差',
        'reasoning': '推理链断裂',
        'decision': '动作决策错误',
        'unknown': '未知错误类型'
    }.get(error_type, '未知错误类型')

    # 构造参考案例部分
    ref_section = ""
    if counterfactuals:
        ref_section = "\n\n【参考成功案例】\n"
        for i, cf in enumerate(counterfactuals[:3]):
            sample = cf.get('sample', {})
            cand_inst = sample.get('gpt_instruction', '')
            cand_actions = sample.get('action', [])[:5]
            cand_action_names = [action_map.get(a, str(a)) for a in cand_actions]
            ref_section += (
                f"案例{i+1}: 指令='{cand_inst[:60]}' "
                f"动作序列={cand_action_names}\n"
            )

    current_action_desc = ""
    if current_action is not None:
        current_action_desc = (
            f"\n当前模型预测动作: {action_map.get(current_action, str(current_action))}"
            f"（此动作在当前高危状态下不安全）"
        )

    prompt = f"""你是一个无人机导航专家，负责在高风险场景下生成安全的脱险决策。

【任务指令】
{instruction}

【当前状态警告】
- 风险评分: {risk_score:.3f}（高危，阈值0.7）
- 诊断错误类型: {error_desc}
{current_action_desc}
{ref_section}

【要求】
请生成一个安全的脱险 CoT 推理过程，格式如下：
<thinking>
1. 当前状态分析：[描述危险情况]
2. 反事实推理：如果继续当前动作，将会[描述后果]；因此必须[描述替代方案]
3. 安全动作选择：[说明为什么选择该动作]
</thinking>
<action>[0-9之间的单个数字]</action>

动作说明: 0=停止, 1=前进, 2=左转, 3=右转, 4=上升, 5=下降, 6=左移, 7=右移, 8=快速前进, 9=超速

注意：必须优先考虑安全，选择能脱离危险的动作。"""

    return prompt


# ==================== 主类 ====================

class CorrectionLoRADataBuilder:
    """
    Correction LoRA 训练数据构造器

    依赖：
    - stage3 训完的 RiskAwareVLA 权重
    - CounterfactualFinder（250K训练集）
    - Qwen API
    """

    def __init__(
        self,
        config: Stage3Config,
        stage3_checkpoint_dir: str,
        output_path: str,
        risk_threshold: float = 0.7,
        qwen_api_base: str = "http://localhost:8000",
        device: str = "cuda:0"
    ):
        self.config = config
        self.output_path = Path(output_path)
        self.risk_threshold = risk_threshold
        self.qwen_api_base = qwen_api_base
        self.device = device

        print(f"\n{'='*60}")
        print("Correction LoRA 数据构造器初始化")
        print(f"{'='*60}")
        print(f"风险阈值: {risk_threshold}")
        print(f"Qwen API: {qwen_api_base}")
        print(f"输出路径: {output_path}")

        # 加载 stage3 模型
        print("\n[1/2] 加载 RiskAwareVLA (stage3 权重)...")
        self.vla = RiskAwareVLA(
            stage2_checkpoint_dir=stage3_checkpoint_dir,
            num_unfrozen_layers=0,  # 推理模式，不训练
            verbose=True
        ).to(device)
        self.vla.eval()

        # 加载 CounterfactualFinder
        print("\n[2/2] 初始化 CounterfactualFinder...")
        self.cf_finder = CounterfactualFinder(verbose=False)

        self.tokenizer = self.vla.base_vla.llm_backbone.tokenizer
        print(f"\n✅ 初始化完成")

    def _score_sample(self, sample: Dict) -> Tuple[float, str, Optional[int]]:
        """
        用 risk_head_1 和 risk_head_2 对一个样本打分

        Returns:
            (overall_risk, error_type, predicted_action)
        """
        instruction = sample.get('gpt_instruction') or sample.get('instruction', '')
        episode_folder = sample.get('image_path') or sample.get('episode_id', '')
        frame_name = sample.get('frame_idx')
        if not frame_name and 'index_list' in sample and sample['index_list']:
            frame_name = sample['index_list'][0]

        # 加载图片
        pixel_values = None
        img_path = None
        if episode_folder and frame_name:
            base_root = Path(self.config.image_base_path)
            folder_path = base_root / episode_folder
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = folder_path / f"{frame_name}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break

        if img_path is None:
            return 0.5, 'unknown', None

        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            tr_img = self.vla.base_vla.vision_backbone.image_transform(img)
            pixel_values = {}
            for k in tr_img.keys():
                combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                pixel_values[k] = combined.unsqueeze(0).to(self.device)
        except Exception as e:
            return 0.5, 'unknown', None

        # 构造 prompt
        if hasattr(self.vla.base_vla, 'get_prompt_builder'):
            pb = self.vla.base_vla.get_prompt_builder()
        else:
            from model.prompt_llama2 import LLaMa2ChatPromptBuilder
            pb = LLaMa2ChatPromptBuilder("prismatic")
        pb.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?"
        )
        prompt_text = pb.get_prompt()
        input_ids = self.tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)

        try:
            with torch.no_grad():
                outputs = self.vla(
                    pixel_values=pixel_values,
                    input_ids=input_ids
                )
            aux = outputs.get('aux_outputs', {})
            overall_risk = aux.get('overall_risk', torch.tensor([[0.5]])
                                   ).item()

            # error_type
            error_type_probs = aux.get('error_type_probs')
            error_type_map_inv = {0: 'perception', 1: 'comprehension',
                                  2: 'reasoning', 3: 'decision'}
            if error_type_probs is not None:
                et_idx = error_type_probs.argmax(dim=-1).item()
                error_type = error_type_map_inv.get(et_idx, 'unknown')
            else:
                error_type = 'unknown'

            # 预测动作（取 logits 里最后一个 token 的 argmax）
            logits = outputs.get('logits')
            predicted_action = None
            if logits is not None:
                next_token = logits[0, -1, :].argmax().item()
                decoded = self.tokenizer.decode([next_token],
                                                skip_special_tokens=True)
                try:
                    predicted_action = int(decoded.strip())
                except Exception:
                    predicted_action = None

            return overall_risk, error_type, predicted_action

        except Exception as e:
            return 0.5, 'unknown', None

    def _build_failure_case(self, sample: Dict, idx: int) -> Dict:
        """把训练样本转成 CounterfactualFinder 需要的格式"""
        episode_folder = sample.get('image_path') or sample.get('episode_id', '')
        env_name = episode_folder.split('/')[0] if '/' in episode_folder else 'unknown'
        pos_list = sample.get('pos', [[0, 0, 0]])
        start_pos = pos_list[0] if pos_list else [0, 0, 0]

        return {
            'sample_id': str(idx),
            'env_name': env_name,
            'start_pos': start_pos,
            'instruction': sample.get('gpt_instruction') or
                           sample.get('instruction', ''),
            'trajectory': {
                'actions': sample.get('action', [])
            }
        }

    def _find_image_path(self, sample: Dict) -> Optional[str]:
        """找到样本对应的图片路径"""
        episode_folder = sample.get('image_path') or sample.get('episode_id', '')
        frame_name = sample.get('frame_idx')
        if not frame_name and 'index_list' in sample and sample['index_list']:
            frame_name = sample['index_list'][0]
        if not episode_folder or not frame_name:
            return None
        base_root = Path(self.config.image_base_path)
        folder_path = base_root / episode_folder
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = folder_path / f"{frame_name}{ext}"
            if test_path.exists():
                return str(test_path)
        return None

    def build(
        self,
        train_data_path: str,
        resume: bool = True
    ) -> List[Dict]:
        """
        主流程：构造 Correction LoRA 训练数据

        Args:
            train_data_path: 1500样本路径
            resume: 是否断点续传

        Returns:
            correction_samples 列表
        """
        print(f"\n{'='*60}")
        print("开始构造 Correction LoRA 训练数据")
        print(f"{'='*60}")

        # 加载训练数据
        with open(train_data_path, 'r') as f:
            train_data = json.load(f)
        print(f"✅ 加载 {len(train_data)} 个样本")

        # 断点续传
        correction_samples = []
        start_idx = 0
        progress_path = self.output_path.parent / "correction_data_progress.json"

        if resume and progress_path.exists():
            try:
                with open(progress_path, 'r') as f:
                    progress = json.load(f)
                start_idx = progress.get('completed', 0)
                if self.output_path.exists():
                    with open(self.output_path, 'r') as f:
                        saved = json.load(f)
                    correction_samples = saved.get('samples', [])
                print(f"🔄 恢复进度，从样本 {start_idx} 继续")
            except Exception as e:
                print(f"⚠️  恢复失败: {e}，从头开始")
                start_idx = 0
                correction_samples = []

        # Step 1: 风险打分，找高危样本
        print(f"\n[Step 1] 风险打分（阈值={self.risk_threshold}）...")
        high_risk_samples = []

        for i in tqdm(range(start_idx, len(train_data)), desc="风险打分"):
            sample = train_data[i]
            risk_score, error_type, pred_action = self._score_sample(sample)

            if risk_score >= self.risk_threshold:
                high_risk_samples.append({
                    'idx': i,
                    'sample': sample,
                    'risk_score': risk_score,
                    'error_type': error_type,
                    'predicted_action': pred_action
                })

        print(f"✅ 找到 {len(high_risk_samples)} 个高危样本 "
              f"({len(high_risk_samples)/len(train_data):.1%})")

        # Step 2: 为每个高危样本生成反事实脱险 CoT
        print(f"\n[Step 2] 生成反事实脱险 CoT（调 Qwen）...")
        qwen_calls = 0
        qwen_failures = 0

        for item in tqdm(high_risk_samples, desc="生成 CoT"):
            idx = item['idx']
            sample = item['sample']
            risk_score = item['risk_score']
            error_type = item['error_type']
            pred_action = item['predicted_action']

            instruction = (sample.get('gpt_instruction') or
                           sample.get('instruction', ''))

            # 找相似成功样本（方案2）
            failure_case = self._build_failure_case(sample, idx)
            cf_result = self.cf_finder.find_counterfactuals(
                failure_case=failure_case,
                top_k=3,
                verbose=False
            )
            counterfactuals = cf_result.get('counterfactuals', [])

            # 构造 prompt
            prompt = build_correction_prompt(
                instruction=instruction,
                risk_score=risk_score,
                error_type=error_type,
                counterfactuals=counterfactuals,
                current_action=pred_action
            )

            # 调 Qwen
            img_path = self._find_image_path(sample)
            qwen_calls += 1
            correction_cot = call_qwen(
                prompt=prompt,
                image_path=img_path,
                api_base=self.qwen_api_base
            )

            if not correction_cot:
                qwen_failures += 1
                continue

            # 从 Qwen 输出里解析 action
            import re
            action_match = re.search(r'<action>(\d+)</action>', correction_cot)
            correction_action = int(action_match.group(1)) if action_match else None

            if correction_action is None or correction_action not in range(10):
                qwen_failures += 1
                continue

            correction_samples.append({
                'sample_idx': idx,
                'instruction': instruction,
                'image_path': str(img_path) if img_path else '',
                'risk_score': risk_score,
                'error_type': error_type,
                'predicted_action': pred_action,
                'correction_cot': correction_cot,
                'correction_action': correction_action,
                'num_counterfactuals': len(counterfactuals),
                'generated_at': datetime.now().isoformat()
            })

            # 每50个保存一次
            if len(correction_samples) % 50 == 0:
                self._save(correction_samples, qwen_calls, qwen_failures)
                with open(progress_path, 'w') as f:
                    json.dump({'completed': idx + 1,
                               'high_risk_found': len(high_risk_samples),
                               'correction_samples': len(correction_samples)},
                              f)

        # 最终保存
        self._save(correction_samples, qwen_calls, qwen_failures)

        print(f"\n{'='*60}")
        print(f"✅ Correction LoRA 训练数据构造完成")
        print(f"   高危样本数: {len(high_risk_samples)}")
        print(f"   成功生成:   {len(correction_samples)}")
        print(f"   Qwen 调用:  {qwen_calls} 次")
        print(f"   失败/跳过:  {qwen_failures} 次")
        print(f"   保存路径:   {self.output_path}")
        print(f"{'='*60}\n")

        return correction_samples

    def _save(self, samples: List[Dict], qwen_calls: int, qwen_failures: int):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_samples': len(samples),
                'qwen_calls': qwen_calls,
                'qwen_failures': qwen_failures,
                'generated_at': datetime.now().isoformat(),
                'samples': samples
            }, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3_ckpt', type=str, required=True,
                        help='Stage3 最终权重目录')
    parser.add_argument('--output', type=str,
                        default='/home/ubuntu/data1/lyy/full_rlds_project-3/'
                                'stage3/correction_lora_train.json')
    parser.add_argument('--risk_threshold', type=float, default=0.7)
    parser.add_argument('--qwen_api', type=str, default='http://localhost:8000')
    parser.add_argument('--no_resume', action='store_true')
    args = parser.parse_args()

    config = Stage3Config()
    builder = CorrectionLoRADataBuilder(
        config=config,
        stage3_checkpoint_dir=args.stage3_ckpt,
        output_path=args.output,
        risk_threshold=args.risk_threshold,
        qwen_api_base=args.qwen_api
    )
    builder.build(
        train_data_path=config.data_path,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
