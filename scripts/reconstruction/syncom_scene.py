

import argparse
from utils.exp_utils import run_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run reconstructions for SynCom-Scene dataset")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to the dataset directory")
    parser.add_argument("--exp_dir", default="exp", type=str, help="Path to the output directory")
    parser.add_argument("--gpus", default=[0], type=int, nargs="+", help="GPU ID to use for training and rendering")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step")
    parser.add_argument("--skip_render", action="store_true", help="Skip rendering step")
    args = parser.parse_args()

    log_dir = f"{args.exp_dir}/logs"
    print(f"Running SynCom-Obj with root directory: {args.root_dir}")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Log directory: {log_dir}")

    names = ["artwall", "attic", "forest", "room"]
    
    cmd_train_template = (
        "python train.py "
        "--iterations 30000 "
        "--lambda_mask 0.0 "
        "--mask_loss_from_iter 30001 "
        "--opacity_prune_threshold 0.05 "
        "-s {root_dir}/{name} "
        "-m {exp_dir}/{name} "
        "--eval"
    )
    
    cmd_render_template = (
        "python render.py "
        "--iteration -1 "
        "-s {root_dir}/{name} "
        "-m {exp_dir}/{name} "
        "--eval "
        "--skip_train "
        "--num_rays 384 "
        "--render_type rf"
    )

    exp_names_list = []
    cmd_str_list = []

    for name in names:

        # clear command string
        cmd_str = ""

        # train
        if not args.skip_train:
            cmd_str += cmd_train_template.format(
                root_dir=args.root_dir,
                exp_dir=args.exp_dir,
                name=name
            )

        # render
        if not args.skip_render:
            if cmd_str:
                cmd_str += " && "
            cmd_str += cmd_render_template.format(
                root_dir=args.root_dir,
                exp_dir=args.exp_dir,
                name=name
            )

        cmd_str_list.append(cmd_str)
        exp_names_list.append(f"{name}")
            
    run_experiment(exp_names_list, cmd_str_list, args.gpus, log_dir)