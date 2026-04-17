"""
RLDS 图片提取脚本 v2
利用 rlds_available_episodes.json 索引，按 vlnv 分组提取
关键：index_list[i] 对应 steps[i+1]（跳过首帧和末帧）
"""
import json
import tensorflow_datasets as tfds
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def main():
    print("=" * 60)
    print("RLDS Image Extraction v2")
    print("=" * 60)

    # ===== 配置 =====
    sampled_json = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/train_sampled_5000.json"
    rlds_index_json = "/home/ubuntu/data1/lyy/full_rlds_project-3/0_sampling/rlds_available_episodes.json"
    rlds_base = Path("/home/ubuntu/data1/wrp/OpenFly-Platform/train/OpenFly-rlds")
    output_base = Path("/home/ubuntu/data1/lyy/full_rlds_project-3/images")

    # 1. 加载采样数据
    print(f"\nLoading sampled data...")
    with open(sampled_json, 'r') as f:
        sampled = json.load(f)
    print(f"  {len(sampled)} episodes to extract")

    # 2. 加载 RLDS 索引 (image_path -> vlnv_name)
    print(f"Loading RLDS index...")
    with open(rlds_index_json, 'r') as f:
        rlds_index = json.load(f)

    # 3. 按 vlnv 分组
    vlnv_groups = defaultdict(dict)
    missing = []
    for ep in sampled:
        path = ep['image_path']
        vlnv = rlds_index.get(path)
        if vlnv:
            vlnv_groups[vlnv][path] = ep
        else:
            missing.append(path)

    if missing:
        print(f"  WARNING: {len(missing)} episodes not in RLDS index")

    print(f"  Grouped into {len(vlnv_groups)} vlnv datasets")
    for vlnv, eps in sorted(vlnv_groups.items()):
        print(f"    {vlnv}: {len(eps)} episodes")

    # 4. 断点恢复
    progress_file = output_base / "extraction_progress.json"
    completed = set()
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            completed = set(json.load(f).get('completed', []))
        print(f"\n  Resuming: {len(completed)} already extracted")

    # 5. 逐 vlnv 提取
    total_images = 0
    total_episodes = 0
    failed = []
    mismatch_warnings = 0

    for vlnv_name in sorted(vlnv_groups.keys()):
        needed = vlnv_groups[vlnv_name]
        todo = {p: ep for p, ep in needed.items() if p not in completed}
        if not todo:
            print(f"\n  {vlnv_name}: all done, skip")
            continue

        dataset_dir = rlds_base / vlnv_name / "1.0.0"
        if not dataset_dir.exists():
            print(f"\n  {vlnv_name}: no 1.0.0 dir, skip")
            failed.extend(todo.keys())
            continue

        print(f"\n  Processing {vlnv_name} ({len(todo)} episodes)...")

        try:
            builder = tfds.builder_from_directory(str(dataset_dir))
            ds = builder.as_dataset(split='train')

            found = 0
            for rlds_ep in tqdm(ds, desc=f"    {vlnv_name}"):
                file_path = rlds_ep['episode_metadata']['file_path'].numpy().decode('utf-8')
                parts = file_path.split('/')
                image_path = '/'.join(parts[-4:]) if len(parts) >= 4 else file_path

                if image_path not in todo:
                    continue

                ep_info = todo[image_path]
                index_list = ep_info.get('index_list', [])
                out_dir = output_base / image_path
                out_dir.mkdir(parents=True, exist_ok=True)

                # ===== 关键：逐帧提取，避免一次性加载全部到内存 =====
                ep_images = 0
                step_iter = rlds_ep['steps'].as_numpy_iterator()

                # 跳过 step[0]（初始状态帧）
                try:
                    next(step_iter)
                except StopIteration:
                    failed.append(image_path)
                    continue

                # step[1] ~ step[-2] 对应 index_list[0] ~ index_list[-1]
                for i, frame_idx in enumerate(index_list):
                    try:
                        step = next(step_iter)
                        img_array = step['observation']['image_1']
                        img = Image.fromarray(img_array)
                        img.save(out_dir / f"{frame_idx}.png")
                        ep_images += 1
                    except StopIteration:
                        # steps 提前用完，说明 diff != 2 的异常 episode
                        if mismatch_warnings < 10:
                            print(f"\n    WARNING: {image_path} steps exhausted at frame {i}/{len(index_list)}")
                            mismatch_warnings += 1
                        break
                    except Exception as e:
                        if mismatch_warnings < 10:
                            print(f"\n    WARNING: frame {frame_idx} extract failed: {e}")
                            mismatch_warnings += 1

                if ep_images > 0:
                    total_images += ep_images
                    total_episodes += 1
                    completed.add(image_path)
                    found += 1
                else:
                    failed.append(image_path)

                if found >= len(todo):
                    break

            # 每个 vlnv 处理完保存进度
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump({'completed': list(completed)}, f)
            print(f"    Extracted {found}/{len(todo)}, total images: {total_images}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            failed.extend(todo.keys())

    # 6. 最终统计
    print(f"\n{'='*60}")
    print(f"Extraction Complete")
    print(f"{'='*60}")
    print(f"  Episodes extracted: {total_episodes}/{len(sampled)}")
    print(f"  Total images: {total_images}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  First 5 failed:")
        for p in failed[:5]:
            print(f"    {p}")
    print(f"  Output: {output_base}")


if __name__ == "__main__":
    main()
