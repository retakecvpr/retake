import os
import os.path as osp
import json
import io
import math
import base64
from PIL import Image
import pandas as pd
from typing import Optional, List

import numpy as np


class BaseDataset:
    def __init__(self, 
                 anno_file: str,
                 processor_kwargs: str
                 ) -> None:
        self.processor_kwargs = processor_kwargs
        # Load annotations
        with open(anno_file, 'r') as F:
            self.annos = json.load(F)
        # Preprocess meta
        for anno in self.annos:
            # NOTE: Pyarrow caching in LLaMA-Factory will raise error
            # for some complicate json data. So dump to jsons.
            if type(anno['meta']) == str:
                anno['meta'] = json.loads(anno['meta'])

    @staticmethod
    def _get_video_sample_extracted_frames(frame_files: List[str], **kwargs) -> int:
        video_fps = kwargs.get("video_fps")
        video_maxlen = kwargs.get("video_maxlen")
        extraction_fps = kwargs.get("video_frame_extraction_fps")
        total_frames = len(frame_files)
        sample_frames = float(total_frames / extraction_fps) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        sample_frames = math.floor(sample_frames)
        return int(sample_frames / 2) * 2

    @staticmethod
    def _preprocess_image(image, **kwargs):
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @staticmethod
    def _pil_image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
        # Create a BytesIO object to hold the image data
        buffered = io.BytesIO()
        
        # Save the image to the BytesIO object in the specified format
        image.save(buffered, format=format)
        
        # Get the byte data from the BytesIO object
        img_byte_data = buffered.getvalue()
        
        # Encode the byte data to base64
        img_base64 = base64.b64encode(img_byte_data).decode('utf-8')
        
        return img_base64

    def __len__(self):
        return len(self.annos)

    def get_video_message(self, video_root: str):
        base64Frames = []
        frame_files = [
            os.path.join(video_root, file) for file in list(sorted(os.listdir(video_root)))
        ]
        total_frames = len(frame_files)
        sample_frames = self._get_video_sample_extracted_frames(frame_files, **self.processor_kwargs)
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx in sample_indices:
                image = Image.open(frame_file)
                resized_image = self._preprocess_image(image, **self.processor_kwargs)
                base64Frames.append(self._pil_image_to_base64(resized_image))

        image_url_list = list(map(lambda x: {"url": f'data:image/jpg;base64,{x}'}, base64Frames))

        return image_url_list

    def __getitem__(self, idx):
        anno = self.annos[idx]

        # NOTE: for vllm in Qwen2VL
        question = anno["messages"][0]["content"].replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>')
        image_url_list = self.get_video_message(anno["videos"][0])
        meta = anno["meta"]
        meta['answer'] = anno["messages"][1]["content"]

        messages = [
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type":"video_url",
                    "video_url": image_url_list
                }
            ]
            }
        ]

        return (idx, messages, meta)


def eval_videomme_results(anno_id2result, anno_id2meta):
    # Load and merge all result files
    anno_id_list, subfield_list, domain_list, duration_list = [], [], [], []
    gt_answer_list, pred_answer_list, correct_list = [], [], []
    for anno_id in anno_id2result:
        pred_answer = anno_id2result[anno_id]
        meta = anno_id2meta[anno_id]
        gt_answer = meta['answer']

        anno_id_list.append(anno_id)
        subfield_list.append(meta['task_type'])
        domain_list.append(meta['domain'])
        duration_list.append(meta['duration'])
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        if gt_answer.lower() == pred_answer.lower():
            correct_list.append(1)
        else:
            correct_list.append(0)

    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'subfield': subfield_list,
        'domain': domain_list,
        'duration': duration_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': correct_list
    })

    # Evaluation
    # Calculate accuracy per subfield
    subfield_accuracy = infer_result_df.groupby('subfield')['correct'].mean()

    # Calculate accuracy per duration
    duration_accuracy = infer_result_df.groupby('duration')['correct'].mean()

    duration_subfield_accuracy = infer_result_df.groupby(['duration', 'subfield'])['correct'].mean()
    final_df = duration_subfield_accuracy.unstack()

    # Overall agregated in duration
    final_df.loc[len(final_df)] = subfield_accuracy
    final_df.index.values[-1] = 'overall'

    # Overall agregated in subfield
    duration_accuracy.loc[3] = duration_accuracy.mean() # NOTE: This is correct because they have the same number of samples
    duration_accuracy.index.values[-1] = 'overall'
    final_df.insert(0, 'overall', duration_accuracy)

    # Reindex the DataFrame
    new_order = ['short', 'medium', 'long', 'overall']
    eval_result_df = final_df.reindex(new_order)
    eval_result_df *= 100 # to percent
    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def eval_mlvu_results(anno_id2result, anno_id2meta):
    # Load and merge all result files
    anno_id_list, question_type_list = [], []
    gt_answer_list, pred_answer_list, correct_list = [], [], []
    for anno_id in anno_id2result:
        meta = anno_id2meta[anno_id]
        pred_answer = anno_id2result[anno_id]
        gt_answer = meta['answer']

        anno_id_list.append(anno_id)
        question_type_list.append(meta['question_type'])
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        if gt_answer.lower() == pred_answer.lower():
            correct_list.append(1)
        else:
            correct_list.append(0)
    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'question_type': question_type_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': correct_list
    })

    # Evaluation
    # Calculate accuracy for each 'question_type'
    accuracy_by_question_type = infer_result_df.groupby('question_type')['correct'].mean() * 100
    accuracy_by_question_type = accuracy_by_question_type.reset_index()
    accuracy_by_question_type.rename(columns={'correct': 'Accuracy'}, inplace=True)

    # Calculate overall accuracy
    overall_accuracy = accuracy_by_question_type['Accuracy'].mean()

    # Add the overall accuracy to the DataFrame
    overall_df = pd.DataFrame({'question_type': ['Overall'], 'Accuracy': [overall_accuracy]})

    # Combine the results
    eval_result_df = pd.concat([accuracy_by_question_type, overall_df], ignore_index=True)
    eval_result_df = eval_result_df.set_index('question_type').transpose()

    new_order = ['Overall', 
                 'Topic Reasoning', 'Anomaly Recognition', 
                 'Needle QA', 'Ego Reasoning', 'Plot QA',
                 'Action Order', 'Action Count']
    eval_result_df = eval_result_df[new_order]

    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def eval_lvbench_results(anno_id2result, anno_id2meta):
    type2correct_list = {}
    anno_id_list = []
    question_type_list = []
    gt_answer_list = []
    pred_answer_list = []
    infer_result_correct_list = []
    for anno_id in anno_id2result:
        pred_answer = anno_id2result[anno_id]
        meta = anno_id2meta[anno_id]
        gt_answer = meta['answer']
        if gt_answer.lower() == pred_answer.lower():
            correct = 1
        else:
            correct = 0

        anno_id_list.append(anno_id)
        question_type_list.append(json.dumps(meta['question_type']))
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        infer_result_correct_list.append(correct)

        for question_type in meta['question_type'] + ['overall']:
            correct_list = type2correct_list.get(question_type, [])
            correct_list.append(correct)
            type2correct_list[question_type] = correct_list

    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'question_type_list': question_type_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': infer_result_correct_list
    })

    for qtype, correct_list in type2correct_list.items():
        type2correct_list[qtype] = [sum(correct_list) / len(correct_list)]
    # type2correct_list['overall'] = sum([v[0] for v in type2correct_list.values()]) / len(type2correct_list)

    eval_result_df = pd.DataFrame(type2correct_list)

    # Reindex the DataFrame
    new_order = ['entity recognition', 'event understanding', 'key information retrieval', 'temporal grounding', 'reasoning', 'summarization', 'overall']
    eval_result_df = eval_result_df[new_order]
    eval_result_df *= 100 # to percent
    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def get_dataset(dataset_name, anno_file, processor_kwargs):
    if dataset_name.lower() in ['videomme', 'mlvu', 'lvbench']:
        return BaseDataset(anno_file, processor_kwargs)
    else:
        print("Error! Dataset not implemented!", dataset_name)
        raise NotImplementedError


def get_eval_methods(dataset_name):
    if dataset_name.lower() == 'videomme':
        return eval_videomme_results
    elif dataset_name.lower() == 'mlvu':
        return eval_mlvu_results
    elif dataset_name.lower() == 'lvbench':
        return eval_lvbench_results
    else:
        print("Error! Evaluation method not implemented!", dataset_name)
        raise NotImplementedError