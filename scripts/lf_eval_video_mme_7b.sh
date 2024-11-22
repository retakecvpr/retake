ts=`date +%Y_%m_%d_%H_%M`


dataset_name='videomme'
anno_file="LLaMA-Factory/data/video_mme/video_mme.json"
configs=(
    configs/videomme/qwen2vl_7b_128k_video_mme_f256_r448.yaml
    configs/videomme/qwen2vl_7b_128k_video_mme_f1024_r448_keyframe-c1_ch2o05.yaml
)

# Loop through each configuration file
for config in "${configs[@]}"; do
    exp_name=$(basename "$config" .yaml)

    # Change directory to LLaMA-Factory
    cd LLaMA-Factory || exit

    # Train using the current config
    llamafactory-cli train "$config" 2>&1 | tee ./logs/${ts}_${exp_name}.log

    # Change back to the previous directory
    cd ..

    # Run the evaluation script
    python inference/eval_lf.py \
        --config_path "LLaMA-Factory/$config" \
        --anno_file $anno_file \
        --dataset_name $dataset_name \
        --work_dir "LLaMA-Factory"
done


dataset_name='mlvu'
anno_file="LLaMA-Factory/data/mlvu/mlvu.json"
configs=(
    configs/mlvu/qwen2vl_7b_128k_mlvu_25fps_f256_r448.yaml
    configs/mlvu/qwen2vl_7b_128k_mlvu_25fps_f1024_r448_keyframe-c1_ch2o05.yaml
)

# Loop through each configuration file
for config in "${configs[@]}"; do
    exp_name=$(basename "$config" .yaml)

    # Change directory to LLaMA-Factory
    cd LLaMA-Factory || exit

    # Train using the current config
    llamafactory-cli train "$config" 2>&1 | tee ./logs/${ts}_${exp_name}.log

    # Change back to the previous directory
    cd ..

    # Run the evaluation script
    python inference/eval_lf.py \
        --config_path "LLaMA-Factory/$config" \
        --anno_file $anno_file \
        --dataset_name $dataset_name \
        --work_dir "LLaMA-Factory"
done


dataset_name='lvbench'
anno_file="LLaMA-Factory/data/lvbench/lvbench.json"
configs=(
    configs/lvbench/qwen2vl_7b_128k_lvbench_val_25fps_f256_r448.yaml
    configs/lvbench/qwen2vl_7b_128k_lvbench_f1024_r448_keyframe-c1_ch2o05.yaml
)

# Loop through each configuration file
for config in "${configs[@]}"; do
    exp_name=$(basename "$config" .yaml)

    # Change directory to LLaMA-Factory
    cd LLaMA-Factory || exit

    # Train using the current config
    llamafactory-cli train "$config" 2>&1 | tee ./logs/${ts}_${exp_name}.log

    # Change back to the previous directory
    cd ..

    # Run the evaluation script
    python inference/eval_lf.py \
        --config_path "LLaMA-Factory/$config" \
        --anno_file $anno_file \
        --dataset_name $dataset_name \
        --work_dir "LLaMA-Factory"
done
