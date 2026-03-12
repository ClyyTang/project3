# ============================================================================
# 配置区域 - 在这里修改主要参数
# ============================================================================

# 输入JSON文件路径（包含要处理的指令数据）
# 默认值: "/home/ubuntu/data1/zx/OpenFly-Platform/tool_ws/src/ins_gen/instructions/train.json"
DEFAULT_INPUT_JSON_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_4500.json"

# 输出JSON文件路径（保存子任务分解结果）
# 默认值: "/home/ubuntu/data1/zx/OpenFly-Platform/tool_ws/src/cot/sub-tasks/sub-tasks.json"
DEFAULT_OUTPUT_JSON_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/subtasks.json"

# GPT API配置文件路径
# 包含模型配置、API密钥、base_url等信息
GPT_API_CONFIG_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/qwen_api_config.json"

# 并发连接数限制（信号量）
# 通常设置为API数量的2倍，根据服务器性能调整
MAX_CONCURRENT_CONNECTIONS = 10

# 批量写入文件的结果数量
# 每收集这么多个成功结果就写入一次文件
BATCH_WRITE_SIZE = 10

# ============================================================================
# 以下是程序代码，一般不需要修改
# ============================================================================

# 导入操作系统接口模块，用于文件和目录操作
import os
# 导入JSON处理模块，用于读写JSON格式数据
import json
# 导入异步IO模块，用于实现异步编程
import asyncio
# 导入异步文件操作模块，用于异步读写文件
import aiofiles
# 导入异步进度条模块，用于显示任务处理进度
from tqdm.asyncio import tqdm
# 导入类型提示，用于代码类型注解
from typing import List, Dict
# 导入OpenAI API客户端
from openai import AsyncOpenAI


# 创建一个异步锁对象，用于保护共享资源的并发访问
file_lock = asyncio.Lock()


# OpenAI客户端类，用于调用GPT API生成子任务
class OpenAIClient:
    # 初始化OpenAI客户端
    def __init__(self, api_key: str, model: str, base_url: str = None):
        # 创建异步OpenAI客户端实例
        if base_url:
            # 如果提供了base_url，使用自定义URL（支持本地模型）
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            # 否则使用默认的OpenAI API
            self.client = AsyncOpenAI(api_key=api_key)
        # 保存模型名称
        self.model = model
        # 初始化token使用统计
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    # 异步调用GPT API，将指令分解为子任务
    async def decompose_instruction(self, instruction: str, actions: List[int]) -> List[str]:
        """
        将一条指令分解为多个子任务
        
        参数:
            instruction: 完整的导航指令
            actions: 动作序列列表
            
        返回:
            子任务列表
        """
        try:
            # 构建GPT对话消息
            messages = [
                {
                    "role": "system",  # 系统消息：定义文本处理任务
                    "content": "You are an assistant proficient in text processing. You need to help me divide one instruction into multiple sub-tasks based on action changes. Provide only the final result as a numbered list without any thinking process or explanation."
                },
                {
                    "role": "user",  # 用户消息：提供要分解的指令数据
                    "content": f"Action sequence: {actions}. Please divide the following instruction into multiple sub-tasks based on the changes in the action sequence. Each sub-task should correspond to a continuous segment of similar actions. Instruction: \"{instruction}\". Provide the result as a numbered list (1., 2., 3., etc.) without any thinking process."
                }
            ]
            
            # 调用OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,  # 控制生成的随机性
                max_tokens=500,  # 限制最大输出token数
            )
            
            # 更新token使用统计
            if response.usage:
                self.total_tokens += response.usage.total_tokens
                self.input_tokens += response.usage.prompt_tokens
                self.output_tokens += response.usage.completion_tokens
            
            # 获取生成的文本
            result = response.choices[0].message.content.strip()
            
            # 解析返回的子任务（假设返回的是带编号的列表格式）
            subtasks = []
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                    # 移除编号前缀（如 "1. ", "- ", "* " 等）
                    cleaned = line.lstrip('0123456789.-*) ').strip()
                    if cleaned:
                        subtasks.append(cleaned)
            
            # 如果没有解析到子任务，直接返回原始结果
            if not subtasks:
                subtasks = [result]
            
            # 返回所有解析出的子任务，不做数量限制
            return subtasks
            
        except Exception as e:
            # 如果出错，打印错误信息并返回空列表
            print(f"Error in decompose_instruction: {str(e)}")
            return []

    # 获取token使用统计
    def get_token_usage(self):
        return {
            'total': self.total_tokens,
            'input': self.input_tokens,
            'output': self.output_tokens
        }


# OpenAI客户端连接池类，用于管理多个API密钥和模型
class OpenAIPool:
    # 初始化连接池
    def __init__(self, configs: List[Dict]):
        # 创建客户端列表
        self.clients = []
        # 遍历所有配置
        for conf in configs:
            # 为每个配置创建一个OpenAI客户端实例
            client = OpenAIClient(
                api_key=conf['key'],  # API密钥
                model=conf['model'],  # 使用的模型名称
                base_url=conf.get('base_url', None)  # 支持本地模型的base_url（可选）
            )
            # 将客户端添加到列表
            self.clients.append(client)
        # 当前使用的客户端索引，用于轮询
        self.index = 0
        # 创建异步锁，保护索引的并发访问
        self.lock = asyncio.Lock()

    # 异步获取一个客户端（轮询方式）
    async def get_client(self) -> OpenAIClient:
        # 使用锁保护，防止并发冲突
        async with self.lock:
            # 获取当前索引对应的客户端
            client = self.clients[self.index]
            # 更新索引，循环使用客户端（轮询算法）
            self.index = (self.index + 1) % len(self.clients)
            # 返回客户端
            return client

    # 获取所有客户端的token使用统计
    def get_tokens(self):
        token_usage = 0  # 总token使用量
        input_token = 0  # 输入token使用量
        output_token = 0  # 输出token使用量
        # 遍历所有客户端
        for client in self.clients:
            # 获取每个客户端的token使用情况
            tokens = client.get_token_usage()
            # 累加总使用量
            token_usage += tokens['total']
            # 累加输入token
            input_token += tokens['input']
            # 累加输出token
            output_token += tokens['output']
        # 返回汇总的token使用情况
        return {'total': token_usage, 'input': input_token, 'output': output_token}


# 异步处理单条指令数据的函数
async def process_instruction(data: Dict, pool: 'OpenAIPool'):
    """
    处理单条指令数据，将其分解为子任务
    
    参数:
        data: 包含image_path、gpt_instruction、action等字段的字典
        pool: OpenAI客户端连接池
        
    返回:
        (状态, 结果数据) 元组
    """
    try:
        # 从连接池获取一个可用的OpenAI客户端
        client = await pool.get_client()
        
        # 提取必要的数据
        image_path = data.get('image_path', '')
        instruction = data.get('gpt_instruction', '')
        actions = data.get('action', [])
        
        # 如果没有指令或动作，跳过处理
        if not instruction or not actions:
            return ('skip', {'path': image_path, 'reason': 'Missing instruction or actions'})
        
        # 调用GPT API分解指令为子任务
        subtasks = await client.decompose_instruction(instruction, actions)
        
        # 如果没有成功分解出子任务
        if not subtasks:
            return ('fail', {'path': image_path, 'error': 'Failed to decompose instruction'})
        
        # 构建子任务字典
        subtasks_dict = {}
        for i, subtask in enumerate(subtasks, 1):
            subtasks_dict[str(i)] = subtask
        
        # 返回成功状态和包含子任务的数据
        return ('success', {
            'image_path': image_path,
            'sub-tasks': subtasks_dict
        })
        
    except Exception as e:
        # 如果处理失败，返回失败状态和错误信息
        return ('fail', {'path': data.get('image_path', 'unknown'), 'error': str(e)})


# 使用信号量限制并发任务数量的包装函数
async def sem_task(task, semaphore):
    # 使用信号量，确保同时运行的任务数不超过限制
    async with semaphore:
        # 执行并返回任务结果
        return await task


# 主函数，程序的入口点
async def main():
    # 导入命令行参数解析模块
    import argparse

    # 创建参数解析器，用于处理命令行参数
    parser = argparse.ArgumentParser(description="Sub-task Generation from Instructions")
    # 添加input参数，用于指定输入的JSON文件路径（可选，有默认值）
    parser.add_argument('-i', '--input', type=str, default=DEFAULT_INPUT_JSON_PATH,
                       help=f"Input JSON path (default: {DEFAULT_INPUT_JSON_PATH})")
    # 添加output参数，用于指定输出的JSON文件路径（可选，有默认值）
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_JSON_PATH,
                       help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON_PATH})")
    # 解析命令行参数
    args = parser.parse_args()

    # 获取输入和输出文件路径
    input_path = args.input
    output_path = args.output

    # 打印处理信息
    print(f"Loading instructions from: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # 打开并读取GPT API配置文件
    with open(GPT_API_CONFIG_PATH, "r") as config_file:
        # 加载API配置信息（包含多个API密钥和模型配置）
        api_configs = json.load(config_file)

    # 使用配置创建OpenAI客户端连接池
    pool = OpenAIPool(api_configs)

    # 打开并读取输入的JSON文件（包含要处理的指令列表）
    with open(input_path, 'r', encoding='utf-8') as f:
        instructions_data = json.load(f)

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 使用配置中的并发连接数创建信号量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONNECTIONS)

    # 为每条指令创建异步处理任务
    tasks = [
        # 使用信号量包装每个任务，控制并发数
        sem_task(process_instruction(data, pool), semaphore)
        for data in instructions_data
    ]

    # 创建空列表用于临时存储结果
    results = []
    # 计数器，记录已收集的结果数量
    count = 0
    # 统计成功、失败和跳过的数量
    success_count = 0
    fail_count = 0
    skip_count = 0

    # 异步打开输出文件，准备写入JSON数据
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as file:
        # 写入JSON数组的开始标记
        await file.write('[')
        # 标记是否是第一个写入的元素（用于控制逗号分隔）
        first = True

        # 使用tqdm显示进度条，遍历所有异步完成的任务
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Instructions:"):
            # 等待任务完成并获取状态和数据
            status, data = await future
            # 如果处理成功
            if status == 'success':
                # 将成功的结果添加到临时列表
                results.append(data)
                # 计数器加1
                count += 1
                success_count += 1
            elif status == 'fail':
                fail_count += 1
                print(f"\nFailed to process: {data.get('path', 'unknown')} - {data.get('error', 'unknown error')}")
            elif status == 'skip':
                skip_count += 1

            # 使用配置中的批量写入大小
            if count >= BATCH_WRITE_SIZE:
                # 如果不是第一批数据，需要在前面加逗号
                if not first:
                    await file.write(',')
                # 将结果列表转换为JSON格式并写入文件
                await file.write(','.join(json.dumps(result, indent=4, ensure_ascii=False) for result in results))
                # 清空结果列表，准备收集下一批
                results.clear()
                # 重置计数器
                count = 0
                # 标记已经不是第一批数据
                first = False

        # 处理剩余的结果（不足BATCH_WRITE_SIZE个的最后一批）
        if results:
            # 如果不是第一批，添加逗号分隔
            if not first:
                await file.write(',')
            # 写入剩余的结果
            await file.write(','.join(json.dumps(result, indent=4, ensure_ascii=False) for result in results))

        # 写入JSON数组的结束标记
        await file.write(']')

    # 从连接池获取token使用情况
    tokens = pool.get_tokens()

    # 打印处理统计信息
    print('\n' + '='*60)
    print('Processing Summary:')
    print(f'  Total instructions: {len(instructions_data)}')
    print(f'  Successfully processed: {success_count}')
    print(f'  Failed: {fail_count}')
    print(f'  Skipped: {skip_count}')
    print('\nToken Usage:')
    print(f'  Total tokens: {tokens["total"]}')
    print(f'  Input tokens: {tokens["input"]}')
    print(f'  Output tokens: {tokens["output"]}')
    print('='*60)


# Python程序的标准入口点判断
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
