"""
训练/测试集划分
5000 episodes → 4500 训练 + 500 测试
保证测试集环境分布均衡
"""
import json
from pathlib import Path
from collections import Counter


def main():
    print("=" * 60)
    print("Train/Test Split")
    print("=" * 60)

    input_path = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_sampled_5000.json"
    train_output = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_4500.json"
    test_output = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/test_500.json"

    with open(input_path, 'r') as f:
        all_data = json.load(f)
    print(f"  Total: {len(all_data)} episodes")

    train_data = all_data[:4500]
    test_data = all_data[4500:]

    print(f"\n  Train: {len(train_data)} episodes")
    print(f"  Test:  {len(test_data)} episodes")

    train_steps = sum(len(ep['action']) for ep in train_data)
    test_steps = sum(len(ep['action']) for ep in test_data)
    print(f"\n  Train steps: {train_steps}")
    print(f"  Test steps:  {test_steps}")

    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Test set environment distribution:")
    test_envs = Counter(ep['image_path'].split('/')[0] for ep in test_data)
    for env, count in sorted(test_envs.items()):
        print(f"    {env}: {count}")

    print(f"\n✅ Saved: {train_output}")
    print(f"✅ Saved: {test_output}")


if __name__ == "__main__":
    main()
