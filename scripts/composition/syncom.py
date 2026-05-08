
import argparse
from utils.exp_utils import run_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SynCom composition script")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to the dataset directory, e.g., D:\\data\\SynCom\\data\\composition")
    parser.add_argument("--exp_dir", default="exp", type=str, help="Path to the output directory")
    parser.add_argument("--gpus", default=[0], type=int, nargs="+", help="GPU ID to use for training and rendering")
    parser.add_argument("--with_trace", action="store_true", default=False, help="Enable trace-based shadow rendering along with SOPs")
    parser.add_argument("--with_ao", action="store_true", default=False, help="Enable object-level SOPs ambient occlusion in PBR shading")
    parser.add_argument("--fps", type=int, default=5, help="Output video fps")
    parser.add_argument("--seed", type=int, default=None, help="Optional override seed for all compositions")
    parser.add_argument("--sops_num", type=int, default=40_000, help="Number of SOPs samples")
    parser.add_argument("--sops_resolution", type=int, default=16, help="SOPs probe resolution")
    args = parser.parse_args()

    output_subdir = "composition_trace" if args.with_trace else "composition"
    log_dir = f"{args.exp_dir}/{output_subdir}/logs"
    print(f"Running SynCom composition with root directory: {args.root_dir}")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Using GPUs: {args.gpus}")
    print(f"With trace rendering: {args.with_trace}")
    print(f"With object AO: {args.with_ao}")

    composition_configs = {
        "artwall_with_bottle": {"label": "indoor", "seed": 0, "latent_scale": 6.0},
        "artwall_with_horse": {"label": "indoor", "seed": 48, "latent_scale": 4.0},
        "artwall_with_kettle": {"label": "indoor", "seed": 8, "latent_scale": 4.0},
        "artwall_with_toy": {"label": "indoor", "seed": 8, "latent_scale": 4.0},
        "attic_with_bottle": {"label": "indoor", "seed": 32, "latent_scale": 10.0},
        "attic_with_horse": {"label": "indoor", "seed": 24, "latent_scale": 5.0},
        "attic_with_kettle": {"label": "indoor", "seed": 24, "latent_scale": 4.0},
        "attic_with_toy": {"label": "indoor", "seed": 40, "latent_scale": 4.0},
        "forest_with_bottle": {"label": "outdoor", "seed": 0, "latent_scale": 4.0},
        "forest_with_horse": {"label": "outdoor", "seed": 48, "latent_scale": 4.0},
        "forest_with_kettle": {"label": "outdoor", "seed": 24, "latent_scale": 4.0},
        "forest_with_toy": {"label": "outdoor", "seed": 0, "latent_scale": 4.0},
        "room_with_bottle": {"label": "indoor", "seed": 56, "latent_scale": 4.0},
        "room_with_horse": {"label": "indoor", "seed": 56, "latent_scale": 4.0},
        "room_with_kettle": {"label": "indoor", "seed": 24, "latent_scale": 4.0},
        "room_with_toy": {"label": "indoor", "seed": 24, "latent_scale": 4.0},
    }
    
    cmd_composition_template = (
        "python composition.py "
        "-s {root_dir}/{composition_name} "
        "--workspace {exp_dir} "
        "--seed {seed} "
        "--num_rays 256 "
        "--latent_scale {latent_scale} "
        "--label {label} "
        "--fps {fps} "
        "--sops_resolution {sops_resolution} "
        "--sops_num {sops_num} "
    )
    if args.with_trace:
        cmd_composition_template += "--with_trace "
    if args.with_ao:
        cmd_composition_template += "--with_ao "

    exp_names_list = []
    cmd_str_list = []
    
    for composition_name, config in composition_configs.items():
        resolved_seed = args.seed if args.seed is not None else config["seed"]
        print(f"{composition_name}: using seed {resolved_seed}")

        # composition
        cmd_str_list.append(cmd_composition_template.format(
            root_dir=args.root_dir,
            exp_dir=args.exp_dir,
            composition_name=composition_name,
            label=config["label"],
            latent_scale=config["latent_scale"],
            fps=args.fps,
            seed=resolved_seed,
            sops_resolution=args.sops_resolution,
            sops_num=args.sops_num
        ))
        exp_names_list.append(f"{composition_name}")
            
    run_experiment(exp_names_list, cmd_str_list, args.gpus, log_dir)
