
import os
import torch
import subprocess
import platform
from concurrent.futures import ProcessPoolExecutor

def get_available_gpus():
    """
    Get the number of available GPUs using PyTorch
    Tries multiple approaches in order of preference:
    
    Returns a list of GPU indices.
    """
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPUs using PyTorch")
        return list(range(gpu_count))
    
    # If all methods fail, return empty list
    print("Warning: No GPUs detected by any method")
    return []

def run_command(command, gpu_id, log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Executing command: {command}\n")
        f.write(f"Using GPU: {gpu_id}\n")
        f.write("=" * 80 + "\n")
        
        # Set CUDA_VISIBLE_DEVICES environment variable
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Use appropriate shell for different platforms
        if platform.system() == "Windows":
            subprocess.run(command, shell=True, stdout=f, stderr=f, env=env, encoding='utf-8', errors='replace')
        else:
            subprocess.run(command, shell=True, stdout=f, stderr=f, env=env)

def run_experiment(exp_name_list, cmd_str_list, gpu_idx_list, log_dir):
    """
    Run a list of commands in parallel, each on a specified GPU.
    
    Args:
        exp_name_list (list): List of experiment names.
        cmd_str_list (list): List of command strings to execute.
        gpu_idx_list (list): List of GPU indices corresponding to each command.
                           If gpu_idx_list is [-1], will auto-detect and use all available GPUs.
        log_dir (str): Directory to save the log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Handle special case: auto-detect GPUs when gpu_idx_list is [-1]
    if len(gpu_idx_list) == 1 and gpu_idx_list[0] == -1:
        available_gpus = get_available_gpus()
        if available_gpus:
            gpu_idx_list = available_gpus
            print(f"Auto-detected {len(available_gpus)} GPUs: {available_gpus}")
        else:
            # Fallback to CPU (no GPU specification)
            gpu_idx_list = [0]  # Use GPU 0 as default
            print("Warning: No GPUs detected, using default GPU 0")

    with ProcessPoolExecutor(max_workers=len(gpu_idx_list)) as executor:
        futures = []
        for i, (exp_name, cmd_str) in enumerate(zip(exp_name_list, cmd_str_list)):
            gpu_idx = gpu_idx_list[i % len(gpu_idx_list)]
            log_file = os.path.join(log_dir, f"{exp_name}.log")
            print(f"Submitting experiment '{exp_name}' on GPU {gpu_idx}")
            futures.append(executor.submit(run_command, cmd_str, gpu_idx, log_file))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()
