import os
import json
import yaml
import argparse

from dataset_utils import get_dataset, get_eval_methods
from infer_eval import trimm_results


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip the line to remove leading/trailing whitespace
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    # Parse the JSON object from the line
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    return data


def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Video MME Evaluation")
    parser.add_argument('--config_path', 
                        type=str, 
                        help="Path to llamafactory train config")
    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='videomme',
                        help="dataset name"),
    parser.add_argument('--anno_file', 
                        type=str, 
                        help="Path to Video-MME LLaMA Factory Format annotation file")
    parser.add_argument('--work_dir', 
                        type=str, 
                        default="LLaMA-Factory",
                        help="Path to working directory of LLaMA Factory train")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.config_path) as F:
        config = yaml.safe_load(F)
    output_dir = os.path.join(args.work_dir, config['output_dir'])

    dataset = get_dataset(dataset_name=args.dataset_name,
                          anno_file=args.anno_file, 
                          processor_kwargs=None)

    # Get inference results
    pred_msg_file = os.path.join(output_dir, "generated_predictions.jsonl")
    pred_messages = load_jsonl(pred_msg_file)
    anno_id2result = {}
    anno_id2meta = {}
    if len(pred_messages) != len(dataset):
        print("Warning! length of predicted messages and dataset not match! %d!=%d" % (
            len(pred_messages), len(dataset)
        ))
    for idx, anno in enumerate(dataset.annos[:len(pred_messages)]):
        meta = anno["meta"]
        meta['answer'] = anno["messages"][1]["content"]
        pred_answer = trimm_results(pred_messages[idx]['predict'])
        anno_id2result[idx] = pred_answer
        anno_id2meta[idx] = meta

    # Evaluate
    eval_func = get_eval_methods(args.dataset_name)
    eval_result_df, infer_result_df = eval_func(anno_id2result, anno_id2meta)

    # Dump inference & evaluation results
    os.makedirs(output_dir, exist_ok=True)
    answer_file = os.path.join(output_dir, "anno_id2result.json")
    infer_res_file = os.path.join(output_dir, "infer_results.csv")
    eval_res_file = os.path.join(output_dir, "eval_results.csv")

    with open(answer_file, 'w') as F:
        json.dump(anno_id2result, F)
    infer_result_df.to_csv(infer_res_file, index=False)
    eval_result_df.to_csv(eval_res_file, index=True)
