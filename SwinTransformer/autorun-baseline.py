#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动评估不同图像尺寸配置下的SwinTransformer模型
并以Markdown表格形式输出结果
"""

import re
import subprocess
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict

# ==================== 配置区域 ====================

# 图像尺寸配置列表: (inputsz, inputsz_w)
img_size_list: List[Tuple[int, int]] = [
    (192, 256),
    (192, 384),
    (192, 512),
    (256, 512),
    (384, 192),
    (512, 384),
    (512, 256),
    # (512, 224),
    # (384, 128),
    # (384, 224),
    # 可以添加更多配置
]

# 路径配置
SAVE_DIR = "/workspace/ssd/models/qwen/viy/SwinTransformer/result"
DATA_PATH = "/media/disk1/models/dataset/imagenet1K"
CONFIG_FILE = "/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
CHECKPOINT_FILE = "/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
# CONFIG_FILE = "/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swin/swin_small_patch4_window7_224.yaml"
# CHECKPOINT_FILE = "/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin_small_patch4_window7_224.pth"

# 硬件配置
CUDA_DEVICES = "0,1,2,3,4,5,6,7"
BATCH_SIZE = 128
NPROC_PER_NODE = 8
MASTER_PORT = 6555

# 模型名称（用于表格显示）
MODEL_NAME = "SwinTransformer-SmallROPE"

# 工作目录
WORK_DIR = "/workspace/ssd/models/qwen/viy/SwinTransformer"

# ==================== 核心逻辑 ====================

# Acc@1 正则表达式
ACC1_RE = re.compile(r"\*\s*Acc@1\s+([0-9]+(?:\.[0-9]+)?)\b")

def run_evaluation(inputsz: int, inputsz_w: int) -> Dict[str, float]:
    """
    运行评估并返回准确率结果
    
    Args:
        inputsz: 输入图像尺寸（高度/宽度）
        inputsz_w: 输入图像宽度尺寸
    
    Returns:
        包含Acc@1的字典
    """
    # 计算WINDOW_SIZE (整数除法)
    window_size = inputsz_w // 32
    
    # 构建保存路径
    save_path = os.path.join(SAVE_DIR, f"{inputsz}_{inputsz_w}")
    os.makedirs(save_path, exist_ok=True)
    
    # 构建命令（使用shell字符串形式，便于处理--opts参数）
    cmd = (
        f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} "
        f"OMP_NUM_THREADS=1 "
        f"python -m torch.distributed.launch "
        f"--nproc_per_node={NPROC_PER_NODE} "
        f"--master_port={MASTER_PORT} "
        f"--nnodes=1 "
        f"--use_env main.py "
        f"--cfg {CONFIG_FILE} "
        f"--resume {CHECKPOINT_FILE} "
        f"--data-path {DATA_PATH} "
        f"--output {save_path} "
        f"--batch-size {BATCH_SIZE} "
        f"--opts DATA.IMG_SIZE {inputsz} DATA.IMG_SIZE_W {inputsz_w} MODEL.SWIN.WINDOW_SIZE {window_size} "
        f"--eval"
    )
    
    print(f"\n{'='*70}")
    print(f"Evaluating Configuration:")
    print(f"  Input Size: {inputsz}")
    print(f"  Input Size W: {inputsz_w}")
    print(f"  Window Size: {window_size}")
    print(f"  Save Path: {save_path}")
    print(f"{'='*70}\n")
    
    try:
        # 执行命令并实时输出
        process = subprocess.Popen(
            cmd,
            cwd=WORK_DIR,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        acc1_value = None
        
        # 实时读取输出并查找Acc@1
        for line in process.stdout:
            output_lines.append(line)
            sys.stdout.write(line)  # 同时输出到控制台
            
            # 尝试解析Acc@1
            match = ACC1_RE.search(line)
            if match and acc1_value is None:
                acc1_value = float(match.group(1))
                print(f"\n✓ Found Acc@1: {acc1_value:.3f}%\n")
        
        process.wait(timeout=3600)  # 1小时超时
        
        if acc1_value is not None:
            return {"acc1": acc1_value}
        else:
            print("\n✗ Failed to parse Acc@1 from output")
            return {"acc1": None}
            
    except subprocess.TimeoutExpired:
        print("\n✗ Evaluation timeout (exceeded 1 hour)")
        if 'process' in locals() and process:
            process.kill()
        return {"acc1": None}
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        return {"acc1": None}


def generate_markdown_table(results: List[Dict]) -> str:
    """
    生成Markdown表格
    
    Args:
        results: 评估结果列表
    
    Returns:
        Markdown表格字符串
    """
    # 表头
    markdown = "| Model | Input Size x Input Size W | Acc | Acc@1 |\n"
    markdown += "|-------|---------------------------|-----|-------|\n"
    
    # 表格内容
    for result in results:
        model = result["model"]
        input_size = result["input_size"]
        input_size_w = result["input_size_w"]
        acc1 = result["acc1"]
        
        # 格式化Acc@1
        acc1_str = f"{acc1:.3f}%" if acc1 is not None else "N/A"
        
        markdown += f"| {model} | {input_size} x {input_size_w} | Acc@1 | {acc1_str} |\n"
    
    return markdown


def print_summary(results: List[Dict]):
    """打印简要摘要"""
    print(f"\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Size':<15} {'Acc@1':<10}")
    print("-" * 70)
    
    for result in results:
        model = result["model"]
        size_str = f"{result['input_size']}x{result['input_size_w']}"
        acc1_str = f"{result['acc1']:.3f}%" if result['acc1'] is not None else "N/A"
        print(f"{model:<25} {size_str:<15} {acc1_str:<10}")
    
    print(f"{'='*70}\n")


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print(f"Starting Auto Evaluation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Configurations: {len(img_size_list)}")
    print(f"{'='*70}\n")
    
    # 存储所有结果
    all_results = []
    
    # 遍历所有配置
    for idx, (inputsz, inputsz_w) in enumerate(img_size_list, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(img_size_list)}] Starting evaluation for {inputsz}x{inputsz_w}")
        print(f"{'='*70}\n")
        
        # 运行评估
        metrics = run_evaluation(inputsz, inputsz_w)
        
        # 保存结果
        result = {
            "model": MODEL_NAME,
            "input_size": inputsz,
            "input_size_w": inputsz_w,
            "acc1": metrics.get("acc1")
        }
        all_results.append(result)
        
        print(f"\n[{idx}/{len(img_size_list)}] Completed: {inputsz}x{inputsz_w}")
        print(f"{'='*70}\n")
    
    # 打印摘要
    print_summary(all_results)
    
    # 生成Markdown表格
    markdown_table = generate_markdown_table(all_results)
    
    # 打印Markdown格式结果
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Markdown Format)")
    print(f"{'='*70}\n")
    print(markdown_table)
    
    # 保存到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(SAVE_DIR, f"evaluation_results_{timestamp}.md")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("# Model Evaluation Results\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Model:** {MODEL_NAME}  \n")
        f.write(f"**Total Configurations:** {len(img_size_list)}  \n\n")
        f.write("## Evaluation Results\n\n")
        f.write(markdown_table)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()