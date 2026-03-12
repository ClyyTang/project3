#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
思维链数据集生成器
整合 gpt_instruction、sub-tasks、current observations 和 actions
生成两张图片之间的思维链推理数据
"""

# ============================================================================
# 配置区域 - 在这里修改主要参数
# ============================================================================

# train.json文件路径（包含gpt_instruction和action）
DEFAULT_TRAIN_JSON_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_4500.json"

# current_observations.json文件路径（包含当前观察）
DEFAULT_CURRENT_OBS_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/observations.json"

# sub-tasks.json文件路径（包含子任务）
DEFAULT_SUBTASKS_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/subtasks.json"

# 输出JSON文件路径
DEFAULT_OUTPUT_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/cot_dataset.json"

# GPT API配置文件路径
GPT_API_CONFIG_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/qwen_api_config.json"

# 并发连接数限制（信号量）
MAX_CONCURRENT_CONNECTIONS = 10

# 批量写入文件的结果数量
BATCH_WRITE_SIZE = 5

# ============================================================================
# 以下是程序代码
# ============================================================================

import os
import json
import asyncio
import aiofiles
import argparse
import logging
from tqdm.asyncio import tqdm
from typing import List, Dict
from openai import AsyncOpenAI


# 动作映射字典
ACTION_DICT = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
    4: "go up",
    5: "go down",
    6: "move left",
    7: "move right",
    8: "move forward",
    9: "move forward"
}


# OpenAI客户端类
class OpenAIClient:
    def __init__(self, api_key: str, model: str, base_url: str = None):
        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def generate_cot_reasoning(
        self, 
        gpt_instruction: str, 
        subtasks: Dict[str, str],
        current_obs: str,
        next_obs: str,
        action: int,
        prev_index: str,
        current_index: str
    ) -> str:
        """
        生成思维链推理内容
        
        参数:
            gpt_instruction: 完整导航指令
            subtasks: 子任务字典
            current_obs: 当前观察
            next_obs: 下一步观察
            action: 当前动作编号
            prev_index: 前一帧索引
            current_index: 当前帧索引
            
        返回:
            生成的思维链文本（包含<thinking>和<next_action>标签）
        """
        try:
            # 构建子任务列表
            subtask_list = "\n".join([f"{k}. {v}" for k, v in subtasks.items()])
            
            # 获取动作描述
            action_name = ACTION_DICT.get(action, "unknown action")
            
            # 构建prompt
            system_prompt = """You are a helpful assistant with robust navigation capabilities, skilled in carefully observing environments and executing tasks based on instructions. You will be given a sequence of navigation tasks. Leverage your expertise in spatial observation and navigation to complete navigation tasks one by one."""
            
            user_prompt = f"""You are working on a navigation task. You will receive a navigation instruction (overall task), subtask information, and current scene observation.

Navigation Instruction (Overall Task):
{gpt_instruction}

Subtasks:
{subtask_list}

Previous Observation (from frame {prev_index}):
{current_obs}

Current Observation (from frame {current_index}):
{next_obs}

First, sum up your historical movement and observation.
Then, you should check how many subtasks you have completed according to these information.
Next, you need to thoroughly analyze your current task and observation to choose your next action, with the explanation why you choose this action.
Meanwhile, the next action should be: {action} ({action_name})

IMPORTANT: In your <thinking></thinking> section, do NOT explicitly mention the action number ({action}). Instead, explain your reasoning naturally using descriptive language (e.g., "I should turn left", "I need to move forward", "I will continue straight", etc.). Only the final action number should appear in <next_action></next_action>.

You should assume that you really observed these information through vision, and do not mention words like 'history context', 'current observation', etc. Use more natural description.
Ensure your reasoning is written between <thinking></thinking>, and place the final chosen action within <next_action></next_action>.
Strictly adhere to the output format and do not output any other information.

## Output Example
<thinking>I began my journey in the kitchen, noticing a white cabinet, then moved through the living room, observing a wooden table and a black chair. The first task is to find a black chair. I have been to the living room and found the black chair. So I have completed the first task. The second task is to go towards glass doors, go forward and enter the white door on the right. However, I have not found the glass doors yet. So I need to find the glass doors now. Currently, I am in the living room, and the patio area is to my left. The glass doors might be in the patio area. So I decide to turn left.</thinking> <next_action>2</next_action>"""

            # 调用API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            # 更新token统计
            self.total_tokens += response.usage.total_tokens
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
            
            # 提取响应内容
            output = response.choices[0].message.content.strip()
            
            return output
            
        except Exception as e:
            logging.error(f"Error generating CoT reasoning: {e}")
            return None

    def get_token_usage(self):
        return {
            'total': self.total_tokens,
            'input': self.input_tokens,
            'output': self.output_tokens
        }


# OpenAI客户端连接池
class OpenAIPool:
    def __init__(self, configs: List[Dict]):
        self.clients = []
        for conf in configs:
            client = OpenAIClient(
                api_key=conf.get('key') or conf.get('api_key'),  # 支持 'key' 和 'api_key'
                model=conf.get('model'),
                base_url=conf.get('base_url')
            )
            self.clients.append(client)
        self.index = 0
        self.lock = asyncio.Lock()

    async def get_client(self) -> OpenAIClient:
        async with self.lock:
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client

    def get_tokens(self):
        token_usage = 0
        input_token = 0
        output_token = 0
        for client in self.clients:
            usage = client.get_token_usage()
            token_usage += usage['total']
            input_token += usage['input']
            output_token += usage['output']
        return {'total': token_usage, 'input': input_token, 'output': output_token}


# 异步处理单个图片对的思维链生成
async def process_single_cot(
    image_path: str,
    gpt_instruction: str,
    subtasks: Dict[str, str],
    current_dict: Dict[str, str],
    index_list: List[str],
    actions: List[int],
    pool: OpenAIPool
):
    """
    处理单个轨迹的所有连续图片对
    
    返回:
        (状态, CoT数据) 或 (状态, 错误信息)
    """
    try:
        cot_list = []
        
        # 遍历所有连续的图片对
        for i in range(len(index_list) - 1):
            prev_index = index_list[i]
            next_index = index_list[i + 1]
            action = actions[i]
            
            # 获取观察
            current_obs = current_dict.get(prev_index, "")
            next_obs = current_dict.get(next_index, "")
            
            if not current_obs or not next_obs:
                logging.warning(f"Missing observation for {prev_index} or {next_index}")
                continue
            
            # 获取客户端
            client = await pool.get_client()
            
            # 生成思维链
            cot_text = await client.generate_cot_reasoning(
                gpt_instruction=gpt_instruction,
                subtasks=subtasks,
                current_obs=current_obs,
                next_obs=next_obs,
                action=action,
                prev_index=prev_index,
                current_index=next_index
            )
            
            if cot_text:
                # 创建键名
                key = f"{prev_index}-{next_index}"
                cot_list.append({key: cot_text})
        
        return ('success', {
            'image_path': image_path,
            'cot': cot_list
        })
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return ('fail', {'path': image_path, 'error': str(e)})


# 使用信号量限制并发
async def sem_task(task, semaphore):
    async with semaphore:
        return await task


# 主函数
async def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought Dataset Generation")
    parser.add_argument('-t', '--train', type=str, default=DEFAULT_TRAIN_JSON_PATH,
                       help=f"Path to train.json (default: {DEFAULT_TRAIN_JSON_PATH})")
    parser.add_argument('-c', '--current', type=str, default=DEFAULT_CURRENT_OBS_PATH,
                       help=f"Path to current_observations.json (default: {DEFAULT_CURRENT_OBS_PATH})")
    parser.add_argument('-s', '--subtasks', type=str, default=DEFAULT_SUBTASKS_PATH,
                       help=f"Path to sub-tasks.json (default: {DEFAULT_SUBTASKS_PATH})")
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_PATH,
                       help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument('--config', type=str, default=GPT_API_CONFIG_PATH,
                       help=f"GPT API config path (default: {GPT_API_CONFIG_PATH})")
    parser.add_argument('-m', '--max-concurrent', type=int, default=MAX_CONCURRENT_CONNECTIONS,
                       help=f"Maximum concurrent connections (default: {MAX_CONCURRENT_CONNECTIONS})")
    
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Loading data files...")
    
    # 读取train.json
    with open(args.train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 读取current_observations.json
    with open(args.current, 'r', encoding='utf-8') as f:
        current_obs_data = json.load(f)
    
    # 读取sub-tasks.json
    with open(args.subtasks, 'r', encoding='utf-8') as f:
        subtasks_data = json.load(f)
    
    # 读取GPT API配置
    with open(args.config, 'r') as f:
        api_configs = json.load(f)
    
    print(f"Loaded {len(train_data)} trajectories")
    
    # 创建索引映射
    current_obs_map = {item['image_path']: item['current'] for item in current_obs_data}
    subtasks_map = {item['image_path']: item['sub-tasks'] for item in subtasks_data}
    
    # ===== 断点恢复：跳过已完成的 episode =====
    completed_episodes = set()
    if os.path.exists(args.output):
        print(f"Checking for existing output...")
        try:
            import json as json2
            with open(args.output, 'r') as ef:
                existing = json2.load(ef)
            completed_episodes = set(item['image_path'] for item in existing)
            print(f"  Found {len(completed_episodes)} completed episodes, will skip them")
        except Exception as e:
            print(f"  Could not parse existing output: {e}, starting fresh")
    
    original_count = len(train_data)
    train_data = [t for t in train_data if t['image_path'] not in completed_episodes]
    print(f"  Remaining: {len(train_data)}/{original_count} episodes")
    # ===== 断点恢复结束 =====
    
    # 创建连接池
    pool = OpenAIPool(api_configs)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 创建信号量
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # 创建任务列表
    tasks = []
    for traj in train_data:
        image_path = traj['image_path']
        gpt_instruction = traj.get('gpt_instruction', '')
        index_list = traj.get('index_list', [])
        actions = traj.get('action', [])
        
        # 获取对应的数据
        current_dict = current_obs_map.get(image_path, {})
        subtasks = subtasks_map.get(image_path, {})
        
        if not current_dict or not subtasks:
            logging.warning(f"Missing data for {image_path}, skipping...")
            continue
        
        # 创建任务
        task = sem_task(
            process_single_cot(
                image_path=image_path,
                gpt_instruction=gpt_instruction,
                subtasks=subtasks,
                current_dict=current_dict,
                index_list=index_list,
                actions=actions,
                pool=pool
            ),
            semaphore
        )
        tasks.append(task)
    
    print(f"Created {len(tasks)} processing tasks")
    
    # 结果列表
    results = []
    success_count = 0
    fail_count = 0
    
    # 异步打开输出文件（追加模式：去掉尾部的]，继续写入）
    if completed_episodes:
        # 续写：去掉末尾的 ] 准备追加
        with open(args.output, 'r') as rf:
            old_content = rf.read()
        old_content = old_content.rstrip().rstrip(']').rstrip()
        with open(args.output, 'w') as wf:
            wf.write(old_content)
    
    async with aiofiles.open(args.output, 'a' if completed_episodes else 'w', encoding='utf-8') as file:
        if not completed_episodes:
            await file.write('[')
        first = not bool(completed_episodes)
        
        # 使用tqdm显示进度
        async for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating CoT"):
            status, data = await task
            
            if status == 'success':
                success_count += 1
                # 写入文件
                if not first:
                    await file.write(',\n')
                else:
                    first = False
                await file.write(json.dumps(data, ensure_ascii=False, indent=4))
                results.append(data)
            else:
                fail_count += 1
                logging.error(f"Failed: {data}")
        
        await file.write('\n]')
    
    # 获取token统计
    tokens = pool.get_tokens()
    
    # 打印统计信息
    print('\n' + '='*60)
    print('Processing Summary:')
    print(f'  Total trajectories: {len(tasks)}')
    print(f'  Successfully processed: {success_count}')
    print(f'  Failed: {fail_count}')
    print('\nToken Usage:')
    print(f'  Total tokens: {tokens["total"]}')
    print(f'  Input tokens: {tokens["input"]}')
    print(f'  Output tokens: {tokens["output"]}')
    print(f'\nResults saved to: {args.output}')
    print('='*60)


if __name__ == "__main__":
    asyncio.run(main())
