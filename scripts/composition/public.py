
import argparse
from utils.exp_utils import run_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SynCom composition script")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to the dataset directory, e.g., D:\\data\\SynCom\\data\\composition")
    parser.add_argument("--exp_dir", default="exp", type=str, help="Path to the output directory")
    parser.add_argument("--gpus", default=[0], type=int, nargs="+", help="GPU ID to use for training and rendering")
    parser.add_argument("--fps", type=int, default=60, help="Output video fps")
    parser.add_argument("--seed", type=int, default=24, help="Unified global seed for envmap generation")
    args = parser.parse_args()

    log_dir = f"{args.exp_dir}/composition/logs"
    print(f"Running SynCom composition with root directory: {args.root_dir}")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Using GPUs: {args.gpus}")

    composition_configs = {
        "caterpillar_with_sculpture": {
            "label": "outdoor",
            "camera_setting": "generate_constrained_cameras",
            "relight_light_mod": 0.1,
            "shadow_light_mod": 0.2,
        },
        "courtroom_with_bell": {
            "label": "indoor",
            "camera_setting": "generate_constrained_cameras",
            "relight_light_mod": 0.3,
            "shadow_light_mod": 0.5,
        },
        "ignatius_with_bull": {
            "label": "outdoor",
            "camera_setting": "generate_cameras",
            "relight_light_mod": 0.3,
            "shadow_light_mod": 0.5,
        },
        "ignatius_with_bull1": {
            "label": "outdoor",
            "camera_setting": "generate_cameras",
            "relight_light_mod": 0.3,
            "shadow_light_mod": 0.5,
        },
        "ignatius_with_bull2": {
            "label": "outdoor",
            "camera_setting": "generate_cameras",
            "relight_light_mod": 0.3,
            "shadow_light_mod": 0.5,
        },
        "ignatius_with_bull3": {
            "label": "outdoor",
            "camera_setting": "generate_cameras",
            "relight_light_mod": 0.3,
            "shadow_light_mod": 0.5,
        },
        "ignatius_with_bull4": {
            "label": "outdoor",
            "camera_setting": "generate_cameras",
            "relight_light_mod": 0.3,
            "shadow_light_mod": 0.5,
        },
        "playground_with_egg": {
            "label": "outdoor",
            "camera_setting": "generate_constrained_cameras",
            "relight_light_mod": 0.05,
            "shadow_light_mod": 0.1,
        },
    }
    
    cmd_composition_template = (
        "python composition.py "
        "-s {root_dir}/{composition_name} "
        "--workspace {exp_dir} "
        "--seed {seed} "
        "--num_rays 256 "
        "--latent_scale 4 "
        "--sops_num 40000 "
        "--label {scene_type} "
        "--fps {fps} "
        "--relight_light_mod {relight_light_mod} "
        "--shadow_light_mod {shadow_light_mod} "
        "--{camera_setting} "
    )

    exp_names_list = []
    cmd_str_list = []
    
    for composition_name, config in composition_configs.items():
        
        # composition
        cmd_str_list.append(cmd_composition_template.format(
            root_dir=args.root_dir,
            exp_dir=args.exp_dir,
            composition_name=composition_name,
            scene_type=config["label"],
            fps=args.fps,
            relight_light_mod=config["relight_light_mod"],
            shadow_light_mod=config["shadow_light_mod"],
            camera_setting=config["camera_setting"],
            seed=args.seed
        ))
        exp_names_list.append(f"{composition_name}")
            
    run_experiment(exp_names_list, cmd_str_list, args.gpus, log_dir)
