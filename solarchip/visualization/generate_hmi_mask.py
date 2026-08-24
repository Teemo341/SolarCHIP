"""
generate_hmi_mask.py —— 从真实 HMI 原始数据统计生成"日面 mask"。

背景：生成模型（尤其 0094->hmi 的 ControlNet）没有约束太阳外围应为 0，
会在盘外生成干扰项。真实 HMI 数据只有日面本体有值，盘外严格为 0。
本脚本统计一批真实 HMI 数据，生成 0/1 mask：
    mask = 1 表示"日面本体"（保留），0 表示"太阳外围"（置零）。

方法：对每个像素取所有样本中 |B| 的最大值，超过阈值的像素记为日面。
由于真实数据盘内无空洞、盘外无孤立点，得到的 mask 天然是一整块实心圆盘。

用法:
    python solarchip/visualization/generate_hmi_mask.py \
        --input_dir logs/sample/pt/hmi/original \
        --output solarchip/visualization/hmi_solar_mask.pt \
        --threshold 1.0
"""
import argparse
import glob
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='从真实 HMI 数据生成日面 mask')
    parser.add_argument('--input_dir', type=str,
                        default='logs/sample/pt/hmi/original',
                        help='真实 HMI 原始 .pt 数据所在目录')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'hmi_solar_mask.pt'),
                        help='mask 保存路径（.pt）')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='|B| 超过该值(高斯)记为日面像素，默认 1.0')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.pt')))
    if not files:
        raise FileNotFoundError(f'{args.input_dir} 下没有 .pt 文件')

    print(f'[mask] 统计 {len(files)} 个文件: {args.input_dir}')
    stack = torch.stack([torch.load(f, weights_only=True) for f in files], dim=0)
    print(f'[mask] 数据形状: {tuple(stack.shape)}, dtype: {stack.dtype}')

    max_abs = stack.abs().max(dim=0).values
    mask = (max_abs > args.threshold).float()
    print(f'[mask] 日面占比: {mask.float().mean():.4f}')

    # 只保留最大连通域，兜底排除个别孤立噪声点
    mask = _keep_largest_component(mask)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(mask, args.output)
    print(f'[mask] 已保存: {args.output} ({tuple(mask.shape)})')


def _keep_largest_component(mask):
    """保留二值 mask 中最大的连通域（日面盘），其余置 0。"""
    mask_np = mask.numpy().astype(np.uint8)
    visited = np.zeros_like(mask_np, dtype=bool)
    best = np.zeros_like(mask_np, dtype=np.uint8)
    best_size = 0
    from collections import deque

    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x] and not visited[y, x]:
                comp = []
                queue = deque([(y, x)])
                visited[y, x] = True
                while queue:
                    cy, cx = queue.popleft()
                    comp.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < mask_np.shape[0] and 0 <= nx < mask_np.shape[1]
                                and mask_np[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                if len(comp) > best_size:
                    best_size = len(comp)
                    best = np.zeros_like(mask_np, dtype=np.uint8)
                    for py, px in comp:
                        best[py, px] = 1
    return torch.from_numpy(best).float()


if __name__ == '__main__':
    main()
