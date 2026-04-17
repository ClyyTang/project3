"""
数据合并脚本
合并 train_sampled.json 和 cot_dataset.json
输出最终的 train_with_cot.json
"""
import json
from pathlib import Path


def load_json(path):
    """加载JSON文件"""
    print(f"📂 加载: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"✅ 加载了 {len(data)} 条数据")
    return data


def merge_cot(train_data, cot_data):
    """
    合并CoT到训练数据
    
    Args:
        train_data: train_sampled.json 的数据
        cot_data: cot_dataset.json 的数据
        
    Returns:
        合并后的数据
    """
    print(f"\n🔗 开始合并数据...")
    
    # 构建CoT索引 {image_path: cot_dict}
    cot_index = {}
    for item in cot_data:
        image_path = item['image_path']
        cot_list = item.get('cot', [])
        
        # 将列表格式转为字典格式
        cot_dict = {}
        for cot_item in cot_list:
            # cot_item 是 {"frame1-frame2": "..."}
            cot_dict.update(cot_item)
        
        cot_index[image_path] = cot_dict
    
    print(f"✅ 构建了 {len(cot_index)} 个CoT索引")
    
    # 合并
    merged = []
    matched = 0
    unmatched = 0
    
    for episode in train_data:
        image_path = episode['image_path']
        
        # 查找对应的CoT
        if image_path in cot_index:
            episode['cot'] = cot_index[image_path]
            matched += 1
        else:
            episode['cot'] = {}
            unmatched += 1
        
        merged.append(episode)
    
    print(f"✅ 合并完成:")
    print(f"  匹配: {matched}")
    print(f"  未匹配: {unmatched}")
    
    return merged


def save_merged_data(data, output_path):
    """保存合并后的数据"""
    print(f"\n💾 保存到: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 统计
    total_cot_pairs = sum(len(ep.get('cot', {})) for ep in data)
    
    print(f"✅ 保存完成")
    print(f"\n📊 统计:")
    print(f"  Episodes: {len(data)}")
    print(f"  总CoT对数: {total_cot_pairs}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据合并脚本")
    print("=" * 60)
    
    # 配置
    train_json = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_4500.json"
    cot_json = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/cot_dataset.json"
    output_json = "/home/ubuntu/data1/lyy/full_rlds_project-3/2_data_merge/train_with_cot_4500.json"
    
    # 1. 加载数据
    train_data = load_json(train_json)
    cot_data = load_json(cot_json)
    
    # 2. 合并
    merged_data = merge_cot(train_data, cot_data)
    
    # 3. 保存
    save_merged_data(merged_data, output_json)
    
    print("\n" + "=" * 60)
    print("✅ 数据合并完成！")
    print("=" * 60)
    print(f"\n最终数据: {output_json}")


if __name__ == "__main__":
    main()