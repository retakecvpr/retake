# Prepare environment
```bash
conda env create -f environment.yaml
conda activate retake
cd LLaMA-Factory
pip install -e .
cd ../transformers
pip install -e .
```


# Prepare data

- Prepare dataset

```bash
# Step 1: Download VideoMME, MLVU, and LVBench from huggingface

# Step 2: Extract frames of videos in 25 fps, remember to replace `videofile_tpl` and `results_dir` in the following scripts
python scripts/utils/frame_extraction.py

# Step 3: Construct the dataset, remember to replace `hf_root`, `absolute_path_to_video_root` and `data_root` in the folloing configs
python scripts/utils/build_lvbench_dataset.py
python scripts/utils/build_mlvu_dataset.py
python scripts/utils/build_videomme_dataset.py

# Step 4: Create a softlink to `LLaMA-Factory/LLaMA-Factory/data`
ln -s ${YOUR_PATH_TO_VIDEOMME} ${PATH_TO_REPO}/LLaMA-Factory/data/videomme
ln -s ${YOUR_PATH_TO_MLVU} ${PATH_TO_REPO}/LLaMA-Factory/data/mlvu
ln -s ${YOUR_PATH_TO_LVBENCH} ${PATH_TO_REPO}/LLaMA-Factory/data/lvbench
```

- Prapare huggingface models
```bash
python scripts/utils/update_qwen2vl_hf_with_yarn.py

ln -s ${Your_path_to}/Qwen2-VL-7B-Instruct ${PATH_TO_REPO}/LLaMA-Factory/saves/baselines/Qwen2-VL-7B-Instruct
```


# Reproduce the results
```bash
bash scripts/lf_eval_video_mme_7b.sh
```