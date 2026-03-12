"""
通用断点续传工具

功能：
1. 定期保存checkpoint
2. 从中断处恢复
3. 支持任意长时间任务
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Callable, Any
import fcntl


class CheckpointManager:
    """Checkpoint管理器"""
    
    def __init__(self, checkpoint_path: str, save_interval: int = 100):
        """
        Args:
            checkpoint_path: checkpoint文件路径
            save_interval: 每隔多少个样本保存一次
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.save_interval = save_interval
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
    def save(self, data: Dict[str, Any]):
        """
        保存checkpoint（原子性写入）
        
        Args:
            data: checkpoint数据，应包含：
                - processed_count: 已处理数量
                - results: 已处理的结果
                - timestamp: 时间戳
        """
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        
        # 添加时间戳
        data['timestamp'] = time.time()
        
        # 写入临时文件
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # 原子性重命名
        temp_path.replace(self.checkpoint_path)
    
    def load(self) -> Dict[str, Any]:
        """
        加载checkpoint
        
        Returns:
            checkpoint数据，如果不存在返回None
        """
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            
            # 打印恢复信息
            if 'timestamp' in data:
                elapsed = time.time() - data['timestamp']
                print(f"📦 从checkpoint恢复:")
                print(f"   已处理: {data.get('processed_count', 0)}")
                print(f"   上次保存: {elapsed/60:.1f}分钟前")
            
            return data
            
        except Exception as e:
            print(f"⚠️  Checkpoint加载失败: {e}")
            return None
    
    def clear(self):
        """清除checkpoint"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


class ResumableProcessor:
    """可恢复的处理器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        save_interval: int = 100,
        auto_save: bool = True
    ):
        """
        Args:
            checkpoint_path: checkpoint路径
            save_interval: 保存间隔
            auto_save: 是否自动保存
        """
        self.ckpt_mgr = CheckpointManager(checkpoint_path, save_interval)
        self.auto_save = auto_save
        
        # 状态
        self.processed_count = 0
        self.results = []
        self.start_time = time.time()
        
        # 加载checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载checkpoint"""
        ckpt = self.ckpt_mgr.load()
        if ckpt:
            self.processed_count = ckpt.get('processed_count', 0)
            self.results = ckpt.get('results', [])
            print(f"✅ 恢复成功，从第 {self.processed_count} 个开始")
    
    def process_batch(
        self,
        data_list: List[Any],
        process_fn: Callable,
        desc: str = "Processing"
    ) -> List[Any]:
        """
        批量处理数据（带断点续传）
        
        Args:
            data_list: 数据列表
            process_fn: 处理函数，接收单个数据项，返回结果
            desc: 进度描述
            
        Returns:
            处理结果列表
        """
        from tqdm import tqdm
        
        total = len(data_list)
        
        # 跳过已处理的
        remaining = data_list[self.processed_count:]
        
        print(f"\n{desc}:")
        print(f"  总数: {total}")
        print(f"  已完成: {self.processed_count}")
        print(f"  剩余: {len(remaining)}")
        
        # 处理
        pbar = tqdm(
            remaining,
            desc=desc,
            initial=self.processed_count,
            total=total
        )
        
        for i, item in enumerate(pbar):
            try:
                # 处理
                result = process_fn(item)
                
                if result is not None:
                    self.results.append(result)
                
                self.processed_count += 1
                
                # 自动保存checkpoint
                if self.auto_save and self.processed_count % self.ckpt_mgr.save_interval == 0:
                    self._save_checkpoint()
                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / elapsed
                    eta = (total - self.processed_count) / rate / 3600
                    pbar.set_postfix({
                        'saved': f'✓',
                        'ETA': f'{eta:.1f}h'
                    })
                
            except Exception as e:
                print(f"\n❌ 处理失败 (index={self.processed_count}): {e}")
                # 继续处理下一个
                self.processed_count += 1
                continue
        
        pbar.close()
        
        # 最终保存
        self._save_checkpoint()
        
        return self.results
    
    def _save_checkpoint(self):
        """保存checkpoint"""
        self.ckpt_mgr.save({
            'processed_count': self.processed_count,
            'results': self.results
        })
    
    def finalize(self, output_path: str):
        """
        完成处理，保存最终结果
        
        Args:
            output_path: 最终输出路径
        """
        print(f"\n💾 保存最终结果: {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 清除checkpoint
        self.ckpt_mgr.clear()
        
        print(f"✅ 完成！总共处理 {self.processed_count} 个")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """演示如何使用"""
    
    print("="*60)
    print("断点续传工具演示")
    print("="*60)
    
    # 1. 创建处理器
    processor = ResumableProcessor(
        checkpoint_path="/tmp/demo_checkpoint.json",
        save_interval=5  # 演示用，每5个保存
    )
    
    # 2. 准备数据
    data_list = list(range(20))
    
    # 3. 定义处理函数
    def process_item(x):
        """处理单个数据项"""
        import time
        time.sleep(0.1)  # 模拟耗时操作
        
        # 模拟偶尔失败
        if x == 7:
            raise ValueError("模拟错误")
        
        return {'input': x, 'output': x * 2}
    
    # 4. 批量处理
    results = processor.process_batch(
        data_list=data_list,
        process_fn=process_item,
        desc="演示处理"
    )
    
    # 5. 保存最终结果
    processor.finalize("/tmp/demo_results.json")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n测试断点续传:")
    print("  1. 运行此脚本")
    print("  2. Ctrl+C 中断")
    print("  3. 再次运行，会从中断处继续")
    print("\nCheckpoint文件: /tmp/demo_checkpoint.json")
    print("最终结果: /tmp/demo_results.json")