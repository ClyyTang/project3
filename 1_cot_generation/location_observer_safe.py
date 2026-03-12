# ============================================================================
# 配置区域 - 在这里修改主要参数
# ============================================================================

DEFAULT_TRAIN_JSON_PATH = "/home/ubuntu/data1/lyy/full_rlds_project/0_sampling/train_sampled_1500.json"
DEFAULT_TRAJ_BASE_PATH = "/home/ubuntu/data1/lyy/full_rlds_project/images"
DEFAULT_OUTPUT_PATH = "/home/ubuntu/data1/lyy/full_rlds_project/1_cot_generation/outputs/observations.json"
GPT_API_CONFIG_PATH = "/home/ubuntu/data1/lyy/full_rlds_project/1_cot_generation/qwen_api_config.json"
CHECKPOINT_PATH = "/home/ubuntu/data1/lyy/full_rlds_project/1_cot_generation/checkpoints/observations_checkpoint.json"
MAX_CONCURRENT_CONNECTIONS = 10
CHECKPOINT_INTERVAL = 10  # 每10个episodes保存一次

import base64
import json
import os
import logging
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
from typing import List, Dict
from mimetypes import guess_type
from openai import AsyncOpenAI
import time


class LocationObserver:
    def __init__(self, api_key, model, base_url=None):
        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.token_usage = 0
        self.input_token = 0
        self.output_token = 0
        
    def update_token_usage(self, response):
        self.token_usage += response.usage.total_tokens
        self.input_token += response.usage.prompt_tokens
        self.output_token += response.usage.completion_tokens

    def local_image_to_data_url(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def extract_landmarks_from_instruction(self, gpt_instruction):
        return gpt_instruction
    
    async def observe_current_location(self, img_path, gpt_instruction):
        try:
            data_url = self.local_image_to_data_url(img_path)
            landmark_context = self.extract_landmarks_from_instruction(gpt_instruction)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an aerial drone navigation assistant. You will be given a first-person image of my movement, you need to accurately identify the user's location."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""This is a first-person image of the current position of the drone. Analyze the scene and identify the iconic buildings or objects mentioned in the navigation context below. Describe what buildings are visible in the current view in one natural sentence.

Navigation context: "{landmark_context}"

Format your answer naturally, for example: "Currently visible is [building description], featuring [features], adjacent to it is [another building], with [features]..." Do not use directional words like left, right, ahead, behind, front, back, etc."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            }
                        ]
                    }
                ],
                extra_body={"include_stop_str_in_output": True}
            )
            
            self.update_token_usage(response)
            output = response.choices[0].message.content
            return output
            
        except Exception as e:
            logging.error(f"Error in observe_current_location: {e}")
            return None

    def get_token_usage(self):
        return {'total': self.token_usage, 'input': self.input_token, 'output': self.output_token}


class OpenAIPool:
    def __init__(self, configs: List[Dict]):
        self.clients = []
        for conf in configs:
            client = LocationObserver(
                api_key=conf['key'],
                model=conf['model'],
                base_url=conf.get('base_url', None)
            )
            self.clients.append(client)
        self.index = 0
        self.lock = asyncio.Lock()

    async def get_client(self) -> LocationObserver:
        async with self.lock:
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client

    def get_tokens(self):
        token_usage = 0
        input_token = 0
        output_token = 0
        for client in self.clients:
            tokens = client.get_token_usage()
            token_usage += tokens['total']
            input_token += tokens['input']
            output_token += tokens['output']
        return {'total': token_usage, 'input': input_token, 'output': output_token}


async def retry_async(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise e


async def process_single_observation(img_index, img_full_path, gpt_instruction, pool: OpenAIPool):
    try:
        if not os.path.exists(img_full_path):
            logging.warning(f"Image file not found: {img_full_path}")
            return ('fail', {img_index: None})
        
        client = await pool.get_client()
        observation = await retry_async(client.observe_current_location, img_full_path, gpt_instruction)
        
        if observation is None:
            logging.warning(f"Observation failed for {img_index}")
            return ('fail', {img_index: None})
        
        return ('success', {img_index: observation})
        
    except Exception as e:
        logging.error(f"Error processing {img_index}: {e}")
        return ('fail', {img_index: None})


async def sem_task(task, semaphore):
    async with semaphore:
        return await task


# ===== 新增：Checkpoint管理 =====
class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    async def save(self, completed_episodes: List[str], results: List[Dict]):
        """保存checkpoint"""
        checkpoint = {
            'timestamp': time.time(),
            'completed_episodes': completed_episodes,
            'completed_count': len(completed_episodes),
            'results': results
        }
        
        # 原子性写入
        temp_path = self.checkpoint_path + '.tmp'
        async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(checkpoint, indent=2, ensure_ascii=False))
        
        # 重命名
        os.replace(temp_path, self.checkpoint_path)
    
    async def load(self):
        """加载checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            return None
        
        try:
            async with aiofiles.open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            checkpoint = json.loads(content)
            
            elapsed = time.time() - checkpoint['timestamp']
            print(f"📦 从checkpoint恢复:")
            print(f"   已完成: {checkpoint['completed_count']} episodes")
            print(f"   上次保存: {elapsed/60:.1f} 分钟前")
            
            return checkpoint
        except Exception as e:
            print(f"⚠️  Checkpoint加载失败: {e}")
            return None
    
    def clear(self):
        """清除checkpoint"""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Current Location Observation Generation (Safe Mode)")
    parser.add_argument('-t', '--train', type=str, default=DEFAULT_TRAIN_JSON_PATH)
    parser.add_argument('-b', '--base', type=str, default=DEFAULT_TRAJ_BASE_PATH)
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('-c', '--config', type=str, default=GPT_API_CONFIG_PATH)
    parser.add_argument('-m', '--max-concurrent', type=int, default=MAX_CONCURRENT_CONNECTIONS)
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 读取配置
    with open(args.config, "r") as config_file:
        api_configs = json.load(config_file)

    pool = OpenAIPool(api_configs)

    # 读取train.json
    with open(args.train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # ===== 加载checkpoint =====
    ckpt_mgr = CheckpointManager(CHECKPOINT_PATH)
    checkpoint = await ckpt_mgr.load()
    
    completed_episodes_set = set()
    all_results = []
    
    if checkpoint:
        completed_episodes_set = set(checkpoint['completed_episodes'])
        all_results = checkpoint['results']
        print(f"✅ 将跳过已完成的episodes")
    
    # 过滤
    train_data = [ep for ep in train_data if ep['image_path'] not in completed_episodes_set]
    print(f"📝 剩余待处理: {len(train_data)} episodes")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # 创建任务
    all_tasks = []
    task_info_list = []
    
    for traj_idx, trajectory in enumerate(train_data):
        image_path = trajectory['image_path']
        gpt_instruction = trajectory['gpt_instruction']
        index_list = trajectory['index_list']
        
        img_folder = os.path.join(args.base, image_path)
        
        for img_idx, img_index in enumerate(index_list):
            img_file = f"{img_index}.png"
            img_full_path = os.path.join(img_folder, img_file)
            
            task = asyncio.create_task(
                sem_task(process_single_observation(img_index, img_full_path, gpt_instruction, pool), semaphore)
            )
            all_tasks.append(task)
            task_info_list.append((traj_idx, img_idx, img_index))

    results = [{} for _ in train_data]
    
    # 初始化输出文件
    if len(all_results) == 0:
        async with aiofiles.open(args.output, 'w', encoding='utf-8') as f:
            await f.write('[\n')
    else:
        # 恢复文件
        async with aiofiles.open(args.output, 'w', encoding='utf-8') as f:
            await f.write('[\n')
            for i, result in enumerate(all_results):
                if i > 0:
                    await f.write(',\n')
                await f.write(json.dumps(result, indent=4, ensure_ascii=False))
    
    file_lock = asyncio.Lock()
    completed_in_this_run = []
    
    # 处理任务
    pending = set(all_tasks)
    with tqdm(total=len(all_tasks), desc="Processing Observations") as pbar:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for completed_task in done:
                task_idx = all_tasks.index(completed_task)
                traj_idx, img_idx, img_index = task_info_list[task_idx]
                
                status, data = await completed_task
                
                if status == 'success':
                    results[traj_idx].update(data)
                
                pbar.update(1)
                
                # 检查episode是否完成
                if len(results[traj_idx]) == len(train_data[traj_idx]['index_list']):
                    async with file_lock:
                        ordered_current = {}
                        for img_index in train_data[traj_idx]['index_list']:
                            if img_index in results[traj_idx]:
                                ordered_current[img_index] = results[traj_idx][img_index]
                        
                        result_entry = {
                            "image_path": train_data[traj_idx]['image_path'],
                            "current": ordered_current
                        }
                        
                        # 写入文件
                        async with aiofiles.open(args.output, 'a', encoding='utf-8') as f:
                            if len(all_results) + len(completed_in_this_run) > 0:
                                await f.write(',\n')
                            await f.write(json.dumps(result_entry, indent=4, ensure_ascii=False))
                        
                        completed_in_this_run.append(train_data[traj_idx]['image_path'])
                        all_results.append(result_entry)
                        results[traj_idx] = {}
                        
                        # 每10个保存checkpoint
                        if len(completed_in_this_run) % CHECKPOINT_INTERVAL == 0:
                            all_completed = list(completed_episodes_set) + completed_in_this_run
                            await ckpt_mgr.save(all_completed, all_results)
                            pbar.set_postfix({'saved': f'✓ ({len(all_completed)})'})

    # 关闭JSON
    async with aiofiles.open(args.output, 'a', encoding='utf-8') as f:
        await f.write('\n]')
    
    # 最终保存checkpoint
    all_completed = list(completed_episodes_set) + completed_in_this_run
    await ckpt_mgr.save(all_completed, all_results)
    
    # 打印统计
    tokens = pool.get_tokens()
    print('\n' + '='*60)
    print('Token Usage Statistics:')
    print('='*60)
    print(f"Total tokens: {tokens['total']}")
    print(f"Input tokens: {tokens['input']}")
    print(f"Output tokens: {tokens['output']}")
    print(f"\nResults saved to: {args.output}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Total completed: {len(all_completed)} episodes")


if __name__ == "__main__":
    asyncio.run(main())