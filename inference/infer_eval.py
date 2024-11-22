import os
import os.path as osp
import json
import re
import io
import math
import base64
import argparse
from tqdm import tqdm
from PIL import Image
import concurrent.futures
from typing import Optional, List
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from openai import OpenAI

from dataset_utils import get_dataset, get_eval_methods


def trimm_results(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def process_message(client, messages, model_name):
    result = client.chat.completions.create(messages=messages, model=model_name, temperature=.0)
    pred_answer = result.choices[0].message.content
    return trimm_results(pred_answer)


def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Video MME Evaluation")
    parser.add_argument('--lf_api_url', 
                        type=str, 
                        help="Url of LLaMA Factory API service")
    parser.add_argument('--model_name_or_path', 
                        type=str, 
                        help="model_name_or_path")
    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='videomme',
                        help="dataset name"),
    parser.add_argument('--anno_file', 
                        type=str, 
                        help="Path to Video-MME LLaMA Factory Format annotation file")
    parser.add_argument('--output', 
                        type=str, 
                        help="Output directory")
    parser.add_argument('--num_workers', 
                        type=int, 
                        help="Number of corcurrent workers")
    parser.add_argument('--video_fps', 
                        type=int, 
                        default=2,
                        help="Video sampling FPS")
    parser.add_argument('--video_maxlen', 
                        type=int, 
                        default=768,
                        help="Maximal sampled frames per video")
    parser.add_argument('--image_resolution', 
                        type=int, 
                        default=128,
                        help="Resolution of maximal side of frame")
    parser.add_argument('--video_frame_extraction_fps', 
                        type=int, 
                        default=5,
                        help="Output directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    client = OpenAI(api_key="0", base_url=args.lf_api_url)

    processor_kwargs = dict(
        video_fps=args.video_fps,
        video_maxlen=args.video_maxlen,
        image_resolution=args.image_resolution,
        video_frame_extraction_fps=args.video_frame_extraction_fps
    )
    dataset = get_dataset(dataset_name=args.dataset_name,
                          anno_file=args.anno_file, 
                          processor_kwargs=processor_kwargs)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers)

    # Inference
    anno_id2result = {}
    anno_id2meta = {}
    for sample in tqdm(dataloader):
        idx, message, meta = sample
        pred_answer = process_message(client, message, args.model_name_or_path)
        anno_id2result[idx] = pred_answer
        anno_id2meta[idx] = meta

    # Evaluate
    eval_func = get_eval_methods(args.dataset_name)
    eval_result_df, infer_result_df = eval_func(anno_id2result, anno_id2meta)

    # Dump inference & evaluation results
    os.makedirs(args.output, exist_ok=True)
    answer_file = os.path.join(args.output, "anno_id2result.json")
    infer_res_file = os.path.join(args.output, "infer_results.csv")
    eval_res_file = os.path.join(args.output, "eval_results.csv")

    with open(answer_file, 'w') as F:
        json.dump(anno_id2result, F)
    infer_result_df.to_csv(infer_res_file, index=False)
    eval_result_df.to_csv(eval_res_file, index=True)
