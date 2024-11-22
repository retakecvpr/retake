import os
import json
import pandas as pd

import glob


hf_root = ".../MLVU"
data_root = ""
absolute_path_to_video_root = '.../MLVU/video_25_fps'

qtype_format_dict = {
    'plotQA': "Plot QA",
    'findNeedle': "Needle QA",
    'ego': "Ego Reasoning",
    'count': "Action Count",
    'order': "Action Order",
    'anomaly_reco': "Anomaly Recognition",
    'topic_reasoning': "Topic Reasoning",
}


data = []
anno_files = glob.glob(os.path.join(hf_root, "MLVU/json/*.json"))
for anno_file in anno_files:
    with open(anno_file, 'r') as F:
        raw_data = json.load(F)
    if '8_sub_scene.json' in anno_file or '9_summary.json' in anno_file:
        continue

    for sample in raw_data:
        if 'candidates' not in sample:
            print("Warning, candidates not found", anno_file)
            continue
        question = sample['question']
        candidates = sample['candidates']
        question = "<video>{question}\nOptions:\nA. {o1}.\nB. {o2}.\nC. {o3}.\nD. {o4}.\nAnswer with the option's letter from the given choices directly.".format(
            question=question, o1=candidates[0], o2=candidates[1], o3=candidates[2], o4=candidates[3]
        )
        answer = None
        for a, cand in zip(['A', 'B', 'C', 'D'], candidates):
            if cand == sample['answer']:
                answer = a
                break
        if answer is None:
            print("Warning! Answer not found!", sample['answer'], candidates)
            continue
        
        question_type = qtype_format_dict[sample['question_type']]

        d = {
            "messages": [
            {
                "content": question,
                "role": "user"
            },
            {
                "content": answer,
                "role": "assistant"
            }
            ],
            "videos": [
                absolute_path_to_video_root.format(
                    typename=os.path.splitext(os.path.basename(anno_file))[0], 
                    videoname=os.path.splitext(sample["video"])[0]
                )
            ],
            "meta": {
                "video": sample["video"],
                "duration": sample["duration"],
                "question_type": question_type,
            }
        }
        data.append(d)


os.makedirs(os.path.join(data_root, "mlvu"), exist_ok=True)
with open(os.path.join(data_root, "mlvu", "mlvu.json"), 'w') as F:
    json.dump(data, F, indent=2)