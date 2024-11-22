import os
import json
import random
import pandas as pd


hf_root = ".../VideoMME"
data_root = ""
absolute_path_to_video_root = '.../VideoMME/video_25_fps'

annos = pd.read_parquet(
    os.path.join(hf_root, 'videomme', 'test-00000-of-00001.parquet')
)

data = []
for idx, row in annos.iterrows():
    question = "<video>%s\nOptions:\n%s\nAnswer with the option's letter from the given choices directly." % (
        row["question"], '\n'.join(row["options"])
    )
    d = {
        "messages": [
        {
            "content": question,
            "role": "user"
        },
        {
            "content": row["answer"],
            "role": "assistant"
        }
        ],
        "videos": [
            os.path.join(absolute_path_to_video_root, row["videoID"])
        ],
        "meta": {
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "duration": row["duration"],
            "domain": row["domain"],
            "sub_category": row["sub_category"],
            "task_type": row["task_type"],
        }
    }
    data.append(d)

os.makedirs(os.path.join(data_root, "video_mme"), exist_ok=True)
with open(os.path.join(data_root, "video_mme", "video_mme.json"), 'w') as F:
    json.dump(data, F, indent=2)
with open(os.path.join(data_root, "video_mme", "video_mme_subset.json"), 'w') as F:
    json.dump(random.sample(data, int(len(data)*0.2)), F, indent=2)