"""
数据采样脚本 v2 (场景多样性优先)
从 train.json (100K) 中筛选 RLDS 可用的 episode，再采样 5000 个
策略：
  1. 只从 RLDS 索引确认有图片的 episode 中选
  2. 按「环境x难度」分层，保证每个组合都有代表
  3. 组合内按动作多样性排序，优先选动作种类丰富的 episode
"""
import json
import random
from pathlib import Path
from collections import defaultdict, Counter


def load_train_json(path):
    print(f"Loading train.json: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"  Total: {len(data)} episodes")
    return data


def load_rlds_index(path):
    print(f"Loading RLDS index: {path}")
    with open(path, 'r') as f:
        index = json.load(f)
    print(f"  Available: {len(index)} episodes")
    return index


def filter_available(train_data, rlds_index):
    available = [ep for ep in train_data if ep['image_path'] in rlds_index]
    print(f"  After filtering: {len(available)} episodes have RLDS images")
    return available


def compute_action_diversity(actions):
    if not actions:
        return 0.0
    unique = set(actions)
    diversity = len(unique) / max(len(actions), 1)
    turns = sum(1 for a in actions if a in [2, 3, 4, 5])
    turn_ratio = turns / max(len(actions), 1)
    return diversity + 0.5 * turn_ratio


def categorize_episodes(data):
    print(f"\nCategorizing episodes...")
    categories = defaultdict(list)
    for ep in data:
        parts = ep['image_path'].split('/')
        if len(parts) >= 3:
            env = parts[0]
            difficulty = parts[2]
            key = f"{env}__{difficulty}"
            categories[key].append(ep)
    print(f"  {len(categories)} categories (env x difficulty)")
    return categories


def diversity_aware_sample(categories, target_total=5000, seed=42):
    random.seed(seed)
    print(f"\nDiversity-aware sampling (target: {target_total})...")

    num_cats = len(categories)
    min_per_cat = 5
    guaranteed = min_per_cat * num_cats
    if guaranteed > target_total:
        min_per_cat = max(1, target_total // num_cats)
        guaranteed = min_per_cat * num_cats
    remaining_quota = target_total - guaranteed
    total_eps = sum(len(eps) for eps in categories.values())

    print(f"  Categories: {num_cats}")
    print(f"  Min per category: {min_per_cat}")
    print(f"  Guaranteed: {guaranteed}")
    print(f"  Remaining quota: {remaining_quota}")

    sampled = []
    cat_counts = {}

    for key, episodes in categories.items():
        scored = [(ep, compute_action_diversity(ep.get('action', []))) for ep in episodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        ratio = len(episodes) / total_eps
        extra = int(remaining_quota * ratio)
        n = min(min_per_cat + extra, len(episodes))
        selected = [ep for ep, _ in scored[:n]]
        sampled.extend(selected)
        cat_counts[key] = len(selected)

    if len(sampled) < target_total:
        deficit = target_total - len(sampled)
        print(f"  Supplementing {deficit} more...")
        used = set(ep['image_path'] for ep in sampled)
        pool = []
        for eps in categories.values():
            for ep in eps:
                if ep['image_path'] not in used:
                    pool.append((ep, compute_action_diversity(ep.get('action', []))))
        pool.sort(key=lambda x: x[1], reverse=True)
        sampled.extend([ep for ep, _ in pool[:deficit]])

    if len(sampled) > target_total:
        random.shuffle(sampled)
        sampled = sampled[:target_total]

    print(f"  Sampled: {len(sampled)} episodes")
    return sampled, cat_counts


def print_stats(sampled):
    print(f"\n{'='*60}")
    print(f"Final Statistics")
    print(f"{'='*60}")
    env_c = Counter()
    diff_c = Counter()
    total_steps = 0
    for ep in sampled:
        parts = ep['image_path'].split('/')
        if len(parts) >= 3:
            env_c[parts[0]] += 1
            diff_c[parts[2]] += 1
        total_steps += len(ep.get('action', []))
    print(f"  Episodes: {len(sampled)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg length: {total_steps/len(sampled):.1f}")
    print(f"\n  [Environments]")
    for e, c in sorted(env_c.items()):
        print(f"    {e}: {c}")
    print(f"\n  [Difficulties]")
    for d, c in sorted(diff_c.items()):
        print(f"    {d}: {c}")


def main():
    print("=" * 60)
    print("Data Sampling v2 (Diversity-Aware)")
    print("=" * 60)

    input_path = "/home/ubuntu/data1/zx/1OpenFly-Platform/OpenFly-Platform/dataset/Annotation/train.json"
    rlds_index_path = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/rlds_available_episodes.json"
    output_path = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_sampled_5000.json"
    target = 5000
    seed = 42

    data = load_train_json(input_path)
    rlds_index = load_rlds_index(rlds_index_path)
    data = filter_available(data, rlds_index)
    categories = categorize_episodes(data)
    sampled, _ = diversity_aware_sample(categories, target_total=target, seed=seed)

    random.seed(seed + 1)
    random.shuffle(sampled)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")

    print_stats(sampled)
    print(f"\n{'='*60}")
    print("Done! Next: python3 extract_images_from_rlds.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
