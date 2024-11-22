import os
import json
import shutil


def create_yarn_scaling_qwen2vl(dir_src, dir_tgt, scaling_factor):
    # Create the target directory if it doesn't exist
    os.makedirs(dir_tgt, exist_ok=True)
    
    # Iterate over all files in the source directory to create symlinks
    for filename in os.listdir(dir_src):
        if filename == "config.json":
            continue

        src_file_path = os.path.join(dir_src, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(src_file_path):
            tgt_file_path = os.path.join(dir_tgt, filename)
            
            # Create a symbolic link in the target directory
            try:
                os.symlink(src_file_path, tgt_file_path)
                print(f"Created symlink for {filename}")
            except FileExistsError:
                print(f"Symlink for {filename} already exists")
            except OSError as e:
                print(f"Failed to create symlink for {filename}: {e}")

    # Copy and edit `config.json` to add YaRN scaling configs
    intput_config_file_path = os.path.join(dir_src, "config.json")
    output_config_file_path = os.path.join(dir_tgt, "config.json")
    with open(intput_config_file_path, 'r') as F:
        config = json.load(F)
    config['rope_scaling']['type'] = 'mrope'
    config['rope_scaling']['factor'] = scaling_factor
    config['rope_scaling']['original_max_position_embeddings'] = config['max_position_embeddings']
    config['rope_scaling']['extrapolation_factor'] = 1
    config['rope_scaling']['attn_factor'] = 1
    config['rope_scaling']['beta_fast'] = 32.0
    config['rope_scaling']['beta_slow'] = 1.0
    config['max_position_embeddings'] *= scaling_factor
    with open(output_config_file_path, 'w') as F:
        json.dump(config, F, indent=2)


if __name__ == "__main__":
    # This code:
    # 1) Create directory `dir_tgt`
    # 2) Create softlink of all files in `dir_src` into `dir_tgt` except `config.json`
    # 3) Copy and edit `config.json` to add YaRN scaling configs

    dir_src = '.../huggingface/Qwen2-VL-7B-Instruct'
    dir_tgt = '.../huggingface/Qwen2-VL-7B-Instruct-128k'
    scaling_factor = 4 # YaRN scaling factor of QWen2
    create_yarn_scaling_qwen2vl(dir_src, dir_tgt, scaling_factor)ß
