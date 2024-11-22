import os
import json
import random
import math
import pandas as pd

import glob



hf_root = ".../LVBench"
data_root = ""
absolute_path_to_video_root = '.../LVBench/video_25_fps'



with open(os.path.join(hf_root, "video_info.meta.jsonl")) as F:
    dataset = [json.loads(line) for line in F.readlines()]

question_type_all = set()

data = []
for video_data in dataset:
    for anno in video_data['qa']:
        question = anno['question'].replace('\n(A)', '\nOptions:\nA.')
        question = question.replace('\n(B)', '\nB.').replace('\n(C)', '\nC.').replace('\n(D)', '\nD.')
        assert '(E)' not in question
        question = f"<video>{question}.\nAnswer with the option's letter from the given choices directly."

        d = {
            "messages": [
            {
                "content": question,
                "role": "user"
            },
            {
                "content": anno['answer'],
                "role": "assistant"
            }
            ],
            "videos": [
                os.path.join(absolute_path_to_video_root, video_data['key'])
            ],
            "meta": {
                "video_id": video_data['key'],
                "uid": anno['uid'],
                "video_type": video_data['type'],
                "question_type": anno['question_type'],
                "time_reference": anno['time_reference'],
            }
        }
        d['meta'].update(video_data['video_info'])
        d['meta'] = json.dumps(d['meta'])
        question_type_all.union(set(anno['question_type']))
        data.append(d)

os.makedirs(os.path.join(data_root, f"lvbench"), exist_ok=True)
with open(os.path.join(data_root, "lvbench", f"lvbench.json"), 'w') as F:
    json.dump(data, F, indent=2)