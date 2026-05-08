
import argparse
from utils.exp_utils import run_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run reconstruction for SynCom-Object dataset")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to the dataset directory")
    parser.add_argument("--exp_dir", default="exp", type=str, help="Path to the output directory")
    parser.add_argument("--gpus", default=[0], type=int, nargs="+", help="GPU ID to use for training and rendering")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step")
    parser.add_argument("--skip_render", action="store_true", help="Skip rendering step")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation step")
    parser.add_argument("--skip_relight", action="store_true", help="Skip relighting step")
    args = parser.parse_args()

    log_dir = f"{args.exp_dir}/logs"
    print(f"Running SynCom-Obj with root directory: {args.root_dir}")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Log directory: {log_dir}")

    names = ["bottle", "horse", "kettle", "toy"]
    
    cmd_train_template = (
        "python train.py "
        "-s {root_dir}/{name} "
        "-m {exp_dir}/{name} "
        "--eval "
        "--sops_num 5_000 "
        "--sops_resolution 16 "
        "--num_rays 128 "
        "--albedo_lr 0.005 "
        "--metallic_lr 0.01 "
        "--roughness_lr 0.01 "
        "--envmap_lr 0.01 "
        "--sops_radiance_lr 0.001 "
        "--sops_alpha_lr 0.002 "
        "--lambda_sops 1.0 "
        "--lambda_light 0.001 "
        "--split_blur "
    )
    
    cmd_render_template = (
        "python render.py "
        "--iteration -1 "
        "-s {root_dir}/{name} "
        "-m {exp_dir}/{name} "
        "--eval "
        "--skip_train "
        "--num_rays 384 "
        "--load_gt "
    )

    cmd_metrics_template = (
        "python metrics.py "
        "-m {exp_dir}/{name} "
        "-t normal albedo pbr "
    )

    cmd_relight_template = (
        "python relight.py "
        "-s {root_dir}/{name} "
        "-m {exp_dir}/{name} "
        "--num_rays 384 "
        "--dataset syncom "
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

        # metrics
        if not args.skip_metrics:
            if cmd_str:
                cmd_str += " && "
            cmd_str += cmd_metrics_template.format(
                root_dir=args.root_dir,
                exp_dir=args.exp_dir,
                name=name
            )

        # relight
        if not args.skip_relight:
            if cmd_str:
                cmd_str += " && "
            cmd_str += cmd_relight_template.format(
                root_dir=args.root_dir,
                exp_dir=args.exp_dir,
                name=name
            )

        cmd_str_list.append(cmd_str)
        exp_names_list.append(f"{name}")
            
    run_experiment(exp_names_list, cmd_str_list, args.gpus, log_dir)