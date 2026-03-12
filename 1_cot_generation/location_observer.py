# ============================================================================
# 配置区域 - 在这里修改主要参数
# ============================================================================

# train.json文件路径
# 包含所有轨迹数据的JSON文件
DEFAULT_TRAIN_JSON_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_4500.json"

# 轨迹图像的基础文件夹路径
# 所有图像文件夹都相对于这个路径
DEFAULT_TRAJ_BASE_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/images"

# 输出JSON文件路径
# 观察结果将保存到这个文件
DEFAULT_OUTPUT_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/observations.json"

# GPT API配置文件路径
# 包含模型配置、API密钥、base_url等信息
GPT_API_CONFIG_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/qwen_api_config.json"

# 并发连接数限制（信号量）
# 通常设置为API数量的2倍，根据服务器性能调整
MAX_CONCURRENT_CONNECTIONS = 10

# 批量写入文件的结果数量
# 每收集这么多个成功结果就写入一次文件
BATCH_WRITE_SIZE = 5

# ============================================================================
# 以下是程序代码，一般不需要修改
# ============================================================================

# 导入base64编码模块，用于将图像转换为base64格式
import base64
# 导入json模块，用于读取和解析JSON数据
import json
# 导入os模块，用于文件路径操作
import os
# 导入logging模块，用于记录日志和错误信息
import logging
# 导入异步IO模块，用于实现异步编程
import asyncio
# 导入异步文件操作模块，用于异步读写文件
import aiofiles
# 导入异步进度条模块，用于显示任务处理进度
from tqdm.asyncio import tqdm
# 导入类型提示，用于代码类型注解
from typing import List, Dict
# 从mimetypes模块导入guess_type函数，用于猜测文件的MIME类型
from mimetypes import guess_type
# 从openai库导入OpenAI客户端类，用于调用OpenAI API
from openai import OpenAI, AsyncOpenAI


# 位置观察客户端类，用于分析无人机当前位置的周围环境
class LocationObserver:
    # 初始化方法
    def __init__(self, api_key, model, base_url=None):
        """
        初始化位置观察客户端（支持本地模型）
        参数:
            api_key: OpenAI API密钥
            model: 使用的模型名称（如gpt-4-vision-preview）
            base_url: 可选，用于连接本地部署的模型服务
        """
        # 如果提供了base_url，创建支持自定义服务地址的客户端
        if base_url:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        else:
            # 否则创建标准OpenAI客户端
            self.client = AsyncOpenAI(
                api_key=api_key,
            )
        # 保存模型名称
        self.model = model
        # 初始化总token使用量计数器
        self.token_usage = 0
        # 初始化输入token使用量计数器
        self.input_token = 0
        # 初始化输出token使用量计数器
        self.output_token = 0
        
    # 更新token使用统计的方法
    def update_token_usage(self, response):
        """
        更新 Token 使用量
        参数:
            response: OpenAI API返回的响应对象
        """
        # 累加总token使用量
        self.token_usage += response.usage.total_tokens
        # 累加输入token（提示词）使用量
        self.input_token += response.usage.prompt_tokens
        # 累加输出token（生成内容）使用量
        self.output_token += response.usage.completion_tokens

    # 将本地图像文件转换为Data URL格式的方法
    def local_image_to_data_url(self, image_path):
        """
        将本地图像转换为 Base64 编码的 Data URL
        这种格式可以直接嵌入到API请求中
        参数:
            image_path: 图像文件的本地路径
        返回:
            data:image/png;base64,... 格式的字符串
        """
        # 根据文件路径猜测MIME类型（如image/png, image/jpeg等）
        mime_type, _ = guess_type(image_path)
        # 如果无法识别类型，使用通用的二进制流类型
        if mime_type is None:
            mime_type = 'application/octet-stream'

        # 以二进制读模式打开图像文件
        with open(image_path, "rb") as image_file:
            # 读取文件内容并进行base64编码，然后解码为UTF-8字符串
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # 返回完整的data URL格式字符串
        return f"data:{mime_type};base64,{base64_encoded_data}"

    # 从gpt_instruction中提取地标信息的方法
    def extract_landmarks_from_instruction(self, gpt_instruction):
        """
        从gpt_instruction中提取地标信息
        参数:
            gpt_instruction: GPT生成的导航指令
        返回:
            地标描述列表
        """
        # 简单的地标提取逻辑：将指令按句子分割，提取建筑物描述
        # 这里可以根据实际需求进行更复杂的NLP处理
        landmarks = []
        
        # 分割指令，查找建筑物描述
        # 通常建筑物描述包含颜色、尺寸、特征等关键词
        instruction_lower = gpt_instruction.lower()
        
        # 提取简化的地标信息用于提示
        return gpt_instruction
    
    # 观察当前位置的方法（异步版本）
    async def observe_current_location(self, img_path, gpt_instruction):
        """
        分析当前位置图像，识别左右前方的标志性物体
        参数:
            img_path: 当前位置的图像路径
            gpt_instruction: 包含路径中地标信息的导航指令
        返回:
            方向性描述字符串
        """
        try:
            # 将图片转换为 Base64 数据URL格式
            data_url = self.local_image_to_data_url(img_path)
            
            # 构建提示词，优先识别gpt_instruction中出现的地标
            landmark_context = self.extract_landmarks_from_instruction(gpt_instruction)

            # 调用 OpenAI 客户端的聊天完成API
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
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                extra_body={"include_stop_str_in_output": True}
            )
            
            # 更新token使用统计
            self.update_token_usage(response)
            # 提取AI的回复内容
            output = response.choices[0].message.content
            # 返回处理后的结果
            return output
            
        except Exception as e:
            # 如果发生错误，记录错误日志
            logging.error(f"Error in observe_current_location: {e}")
            # 返回None表示观察失败
            return None





    # 获取token使用统计的方法
    def get_token_usage(self):
        """
        获取当前 Token 使用量
        返回:
            字典，包含总量、输入token和输出token的统计
        """
        return {
            'total': self.token_usage,
            'input': self.input_token,
            'output': self.output_token
        }


# OpenAI客户端连接池类，用于管理多个API密钥和模型
class OpenAIPool:
    # 初始化连接池
    def __init__(self, configs: List[Dict]):
        # 创建客户端列表
        self.clients = []
        # 遍历所有配置
        for conf in configs:
            # 为每个配置创建一个LocationObserver实例
            client = LocationObserver(
                api_key=conf['key'],
                model=conf['model'],
                base_url=conf.get('base_url', None)
            )
            # 将客户端添加到列表
            self.clients.append(client)
        # 当前使用的客户端索引，用于轮询
        self.index = 0
        # 创建异步锁，保护索引的并发访问
        self.lock = asyncio.Lock()

    # 异步获取一个客户端（轮询方式）
    async def get_client(self) -> LocationObserver:
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
        token_usage = 0
        input_token = 0
        output_token = 0
        # 遍历所有客户端
        for client in self.clients:
            # 获取每个客户端的token使用情况
            tokens = client.get_token_usage()
            # 累加各项统计
            token_usage += tokens['total']
            input_token += tokens['input']
            output_token += tokens['output']
        # 返回汇总的token使用情况
        return {'total': token_usage, 'input': input_token, 'output': output_token}


# 异步重试函数，用于在失败时自动重试
async def retry_async(func, *args, retries=3, delay=2, **kwargs):
    # 循环尝试指定次数
    for attempt in range(retries):
        try:
            # 直接await异步函数
            return await func(*args, **kwargs)
        except Exception as e:
            # 如果不是最后一次尝试
            if attempt < retries - 1:
                # 等待指定的延迟时间后重试
                await asyncio.sleep(delay)
            else:
                # 最后一次尝试失败则抛出异常
                raise e


# 异步处理单个图像的函数
async def process_single_observation(img_index, img_full_path, gpt_instruction, pool: OpenAIPool):
    """
    异步处理单个图像的观察任务
    参数:
        img_index: 图像索引名称
        img_full_path: 图像完整路径
        gpt_instruction: GPT导航指令
        pool: OpenAI客户端连接池
    返回:
        (状态, {索引: 观察结果}) 或 (状态, {索引: 错误信息})
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(img_full_path):
            logging.warning(f"Image file not found: {img_full_path}")
            return ('fail', {img_index: None})
        
        # 从连接池获取一个可用的客户端
        client = await pool.get_client()
        
        # 调用观察方法处理图像
        observation = await retry_async(
            client.observe_current_location,
            img_full_path,
            gpt_instruction
        )
        
        # 检查观察结果
        if observation is None:
            logging.warning(f"Observation failed for {img_index}")
            return ('fail', {img_index: None})
        
        # 返回成功状态和结果
        return ('success', {img_index: observation})
        
    except Exception as e:
        # 如果处理失败，记录错误并返回失败状态
        logging.error(f"Error processing {img_index}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return ('fail', {img_index: None})


# 使用信号量限制并发任务数量的包装函数
async def sem_task(task, semaphore):
    # 使用信号量，确保同时运行的任务数不超过限制
    async with semaphore:
        # 执行并返回任务结果
        return await task


# 主函数
async def main():
    # 导入命令行参数解析模块
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Current Location Observation Generation")
    parser.add_argument('-t', '--train', type=str, 
                       default=DEFAULT_TRAIN_JSON_PATH,
                       help=f"Path to train.json file (default: {DEFAULT_TRAIN_JSON_PATH})")
    parser.add_argument('-b', '--base', type=str,
                       default=DEFAULT_TRAJ_BASE_PATH,
                       help=f"Base path for trajectory images (default: {DEFAULT_TRAJ_BASE_PATH})")
    parser.add_argument('-o', '--output', type=str,
                       default=DEFAULT_OUTPUT_PATH,
                       help=f"Output JSON file path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument('-c', '--config', type=str,
                       default=GPT_API_CONFIG_PATH,
                       help=f"GPT API config file path (default: {GPT_API_CONFIG_PATH})")
    parser.add_argument('-m', '--max-concurrent', type=int, default=MAX_CONCURRENT_CONNECTIONS,
                       help=f"Maximum concurrent connections (default: {MAX_CONCURRENT_CONNECTIONS})")
    parser.add_argument('-w', '--batch-write', type=int, default=BATCH_WRITE_SIZE,
                       help=f"Batch write size (default: {BATCH_WRITE_SIZE})")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 读取GPT API配置文件
    with open(args.config, "r") as config_file:
        api_configs = json.load(config_file)

    # 创建OpenAI客户端连接池
    pool = OpenAIPool(api_configs)

    # 读取train.json文件
    with open(args.train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    

    # ===== 新增：加载已完成的episodes =====
    completed_episodes = set()
    if os.path.exists(args.output):
        print(f"📦 检测到已有输出文件，加载已完成的episodes...")
        with open(args.output, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                for item in existing_data:
                    completed_episodes.add(item['image_path'])
                print(f"✅ 已完成 {len(completed_episodes)} 个episodes，将跳过")
            except:
                print("⚠️  无法解析现有文件，将从头开始")
    
    # 过滤掉已完成的
    train_data = [ep for ep in train_data if ep['image_path'] not in completed_episodes]
    print(f"📝 剩余待处理: {len(train_data)} 个episodes")
    # ===== 新增结束 =====


    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # 创建所有异步任务,按轨迹组织
    all_tasks = []
    task_info_list = []  # 保存每个任务对应的信息 [(traj_idx, img_idx, img_index), ...]
    
    for traj_idx, trajectory in enumerate(train_data):
        image_path = trajectory['image_path']
        gpt_instruction = trajectory['gpt_instruction']
        index_list = trajectory['index_list']
        
        # 构建完整的图像文件夹路径
        # image_path已经包含相对路径，直接和base拼接
        
        img_folder = os.path.join(args.base, image_path)
        
        # 为该轨迹的所有图像创建任务
        for img_idx, img_index in enumerate(index_list):
            img_file = f"{img_index}.png"
            img_full_path = os.path.join(img_folder, img_file)
            
            # 创建异步任务
            task = asyncio.create_task(
                sem_task(
                    process_single_observation(img_index, img_full_path, gpt_instruction, pool),
                    semaphore
                )
            )
            all_tasks.append(task)
            # 保存任务对应的信息,顺序和 all_tasks 一致
            task_info_list.append((traj_idx, img_idx, img_index))

    # 用于存储结果:results[traj_idx][img_index] = observation
    results = [{} for _ in train_data]

    # 初始化输出JSON文件
    if len(completed_episodes) == 0:
        # 第一次运行，创建新文件
        async with aiofiles.open(args.output, 'w', encoding='utf-8') as f:
            await f.write('[\n')
    else:
        # 续传，删除末尾的']'准备追加
        async with aiofiles.open(args.output, 'r', encoding='utf-8') as f:
            content = await f.read()
        content = content.rstrip().rstrip(']').rstrip()
        async with aiofiles.open(args.output, 'w', encoding='utf-8') as f:
            await f.write(content)




    # 创建文件锁
    file_lock = asyncio.Lock()
    
    # 使用进度条和 asyncio.wait 处理所有任务
    pending = set(all_tasks)
    with tqdm(total=len(all_tasks), desc="Processing Observations") as pbar:
        while pending:
            # 等待任何任务完成
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for completed_task in done:
                # 找到完成任务在 all_tasks 中的索引
                task_idx = all_tasks.index(completed_task)
                traj_idx, img_idx, img_index = task_info_list[task_idx]
                
                # 获取任务结果
                status, data = await completed_task
                
                if status == 'success':
                    # 将观察结果添加到对应轨迹
                    results[traj_idx].update(data)
                
                pbar.update(1)
                # 检查该episode是否所有图片都完成
                if len(results[traj_idx]) == len(train_data[traj_idx]['index_list']):
                    # 该episode完成，立即写入
                    async with file_lock:
                        # 按index_list顺序整理
                        ordered_current = {}
                        for img_index in train_data[traj_idx]['index_list']:
                            if img_index in results[traj_idx]:
                                ordered_current[img_index] = results[traj_idx][img_index]
                        
                        result_entry = {
                            "image_path": train_data[traj_idx]['image_path'],
                            "current": ordered_current
                        }
                        

                        async with aiofiles.open(args.output, 'a', encoding='utf-8') as f:
                                # 续传时总是先写逗号
                                await f.write(',\n')
                                await f.write(json.dumps(result_entry, indent=4, ensure_ascii=False))
                                
                        # 清空已写入的结果释放内存
                        results[traj_idx] = {}


    # 打印token使用统计
    tokens = pool.get_tokens()
    print('\n' + '=' * 60)
    print('Token Usage Statistics:')
    print('=' * 60)
    print(f"Total tokens: {tokens['total']}")
    print(f"Input tokens: {tokens['input']}")
    print(f"Output tokens: {tokens['output']}")

    # 关闭JSON数组
    async with aiofiles.open(args.output, 'a', encoding='utf-8') as f:
        await f.write('\n]')
        
    print(f"\nResults saved to: {args.output}")


# Python程序的标准入口点判断
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
