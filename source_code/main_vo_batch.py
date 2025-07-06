 #!/usr/bin/env -S python3 -O

"""
Unified Visual Odometry Benchmark Runner for KITTI and EuRoC datasets.

Set the DATASET variable at the top of the file to "kitti", "euroc", or "both"
to select which pipeline(s) to run. 
"""

import sys
import os
import time
import yaml
import numpy as np
import gc
import torch

# Add thirdparty/relative paths if needed
sys.path.append('/home/ws-rtx/Documents/Projects/pyslam/thirdparty/mast3r/dust3r/croco/models/curope')

# PySLAM modules
from config import Config
from visual_odometry import VisualOdometryEducational
from visual_odometry_rgbd import VisualOdometryRgbdTensor
from camera import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory, SensorType
from feature_tracker import feature_tracker_factory
from feature_tracker_configs import FeatureTrackerConfigs

# ========== SELECT WHICH DATASET TO RUN ==========
# Possible values: "kitti", "euroc", "both"
DATASET = "both"
# =================================================

# ----- Shared tracker configurations -----
TRACKER_CONFIGS = {
    "AKAZE": FeatureTrackerConfigs.AKAZE,
    "ALIKED": FeatureTrackerConfigs.ALIKED,
    "BRISK": FeatureTrackerConfigs.BRISK,
    "BRISK_TFEAT": FeatureTrackerConfigs.BRISK_TFEAT,
    "CONTEXTDESC": FeatureTrackerConfigs.CONTEXTDESC,  
    "D2NET": FeatureTrackerConfigs.D2NET,
    "DISK": FeatureTrackerConfigs.DISK,
    "FAST_FREAK": FeatureTrackerConfigs.FAST_FREAK,
    "FAST_ORB": FeatureTrackerConfigs.FAST_ORB,
    "KAZE": FeatureTrackerConfigs.KAZE,
    "KEYNET": FeatureTrackerConfigs.KEYNET,
    "KEYNETAFFNETHARDNET": FeatureTrackerConfigs.KEYNETAFFNETHARDNET,
    "LFNET": FeatureTrackerConfigs.LFNET,
    "LIGHTGLUE": FeatureTrackerConfigs.LIGHTGLUE,
    "LIGHTGLUE_ALIKED": FeatureTrackerConfigs.LIGHTGLUE_ALIKED,
    "LIGHTGLUE_DISK": FeatureTrackerConfigs.LIGHTGLUE_DISK,
    "LIGHTGLUESIFT": FeatureTrackerConfigs.LIGHTGLUESIFT,
    "LK_FAST": FeatureTrackerConfigs.LK_FAST,
    "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
    "ORB": FeatureTrackerConfigs.ORB,
    "ORB2": FeatureTrackerConfigs.ORB2,
    "ORB2_BEBLID": FeatureTrackerConfigs.ORB2_BEBLID,
    "ORB2_FREAK": FeatureTrackerConfigs.ORB2_FREAK,
    "ORB2_HARDNET": FeatureTrackerConfigs.ORB2_HARDNET,
    "ORB2_L2NET": FeatureTrackerConfigs.ORB2_L2NET,
    "ORB2_SOSNET": FeatureTrackerConfigs.ORB2_SOSNET,
    "R2D2": FeatureTrackerConfigs.R2D2,
    "ROOT_SIFT": FeatureTrackerConfigs.ROOT_SIFT,
    "SHI_TOMASI_FREAK": FeatureTrackerConfigs.SHI_TOMASI_FREAK,
    "SHI_TOMASI_ORB": FeatureTrackerConfigs.SHI_TOMASI_ORB,
    "SIFT": FeatureTrackerConfigs.SIFT,
    "SUPERPOINT": FeatureTrackerConfigs.SUPERPOINT,
    "XFEAT": FeatureTrackerConfigs.XFEAT,
    "XFEAT_LIGHTGLUE": FeatureTrackerConfigs.XFEAT_LIGHTGLUE,
    "XFEAT_XFEAT": FeatureTrackerConfigs.XFEAT_XFEAT,
    "ALIKED_FLANN": FeatureTrackerConfigs.ALIKED_FLANN,
    "BRISK_TFEAT_FLANN": FeatureTrackerConfigs.BRISK_TFEAT_FLANN,
    "CONTEXTDESC_FLANN": FeatureTrackerConfigs.CONTEXTDESC_FLANN,
    "D2NET_FLANN": FeatureTrackerConfigs.D2NET_FLANN,
    "KEYNET_FLANN": FeatureTrackerConfigs.KEYNET_FLANN,
    "KEYNETAFFNETHARDNET_FLANN": FeatureTrackerConfigs.KEYNETAFFNETHARDNET_FLANN,
    "LFNET_FLANN": FeatureTrackerConfigs.LFNET_FLANN,
    "ORB2_BEBLID_FLANN": FeatureTrackerConfigs.ORB2_BEBLID_FLANN,
    "ORB2_FREAK_FLANN": FeatureTrackerConfigs.ORB2_FREAK_FLANN,
    "ORB2_HARDNET_FLANN": FeatureTrackerConfigs.ORB2_HARDNET_FLANN,
    "ORB2_L2NET_FLANN": FeatureTrackerConfigs.ORB2_L2NET_FLANN,
    "ORB2_SOSNET_FLANN": FeatureTrackerConfigs.ORB2_SOSNET_FLANN,
    "ROOT_SIFT_FLANN": FeatureTrackerConfigs.ROOT_SIFT_FLANN,
    "SIFT_FLANN": FeatureTrackerConfigs.SIFT_FLANN,
    "SUPERPOINT_FLANN": FeatureTrackerConfigs.SUPERPOINT_FLANN,
    "XFEAT_FLANN": FeatureTrackerConfigs.XFEAT_FLANN,
    "R2D2_FLANN": FeatureTrackerConfigs.R2D2_FLANN,
}

# ----- KITTI settings -----
KITTI_PATH = "/media/shared/KITTI/dataset_COLOR/"
KITTI_GT_DIR = os.path.join(KITTI_PATH, "poses")
KITTI_OUTPUT_DIR = "/home/ws-rtx/Documents/Projects/pyslam/results_kitti"
KITTI_CONFIG_FILE = "/home/ws-rtx/Documents/Projects/pyslam/config.yaml"
KITTI_SEQUENCES = [
    ("00", "settings/KITTI00-02.yaml"),
    ("01", "settings/KITTI00-02.yaml"),
    ("02", "settings/KITTI00-02.yaml"),
    ("04", "settings/KITTI04-12.yaml"),
    ("05", "settings/KITTI04-12.yaml"),
    ("06", "settings/KITTI04-12.yaml"),
    ("07", "settings/KITTI04-12.yaml"),
    ("08", "settings/KITTI04-12.yaml"),
    ("09", "settings/KITTI04-12.yaml"),
    ("10", "settings/KITTI04-12.yaml"),
]
def load_groundtruth_poses_kitti(sequence_id):
    """Loads ground truth positions for a KITTI sequence."""
    gt_file = os.path.join(KITTI_GT_DIR, f"{sequence_id}.txt")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    poses = np.loadtxt(gt_file, usecols=[3, 7, 11])
    return poses

def run_kitti_pipeline():
    """Runs visual odometry benchmark on the KITTI dataset."""
    print("\n==== Running KITTI pipeline ====")
    for tracker_name, tracker_config in TRACKER_CONFIGS.items():
        tracker_output_dir = os.path.join(KITTI_OUTPUT_DIR, tracker_name)
        os.makedirs(tracker_output_dir, exist_ok=True)
        runtime_file = os.path.join(tracker_output_dir, 'runtimes.txt')

        for seq, settings_file in KITTI_SEQUENCES:
            print(f"\n[Sequence: KITTI {seq} | Tracker: {tracker_name}]")
            traj_file = os.path.join(tracker_output_dir, f"{seq}.txt")
            if os.path.exists(traj_file):
                print(f"  -> Output already exists, skipping: {traj_file}")
                continue

            # Update KITTI config yaml
            with open(KITTI_CONFIG_FILE, "r") as file:
                config_yaml = yaml.safe_load(file)
            config_yaml["KITTI_DATASET"]["name"] = seq
            config_yaml["KITTI_DATASET"]["settings"] = settings_file
            config_yaml["KITTI_DATASET"]["base_path"] = KITTI_PATH
            with open(KITTI_CONFIG_FILE, "w") as file:
                yaml.dump(config_yaml, file, default_flow_style=False)

            start_time = time.time()
            # Initialize modules
            config = Config()
            dataset = dataset_factory(config)
            groundtruth = groundtruth_factory(config.dataset_settings)
            cam = PinholeCamera(config)
            num_features = config.num_features_to_extract if config.num_features_to_extract > 0 else 2000
            tracker_config['num_features'] = num_features

            if tracker_name.upper() == "LIGHTGLUESIFT":
                if "matcher_conf" in tracker_config:
                    tracker_config["matcher_conf"]["add_scale_ori"] = False

            feature_tracker = feature_tracker_factory(**tracker_config)
            if hasattr(feature_tracker, 'detector') and hasattr(feature_tracker.detector, 'net'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                feature_tracker.detector.net.eval()
                feature_tracker.detector.net.to(device)

            if dataset.sensor_type == SensorType.RGBD:
                vo = VisualOdometryRgbdTensor(cam, groundtruth)
            else:
                vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)

            gt_poses = load_groundtruth_poses_kitti(seq)
            img_id = 0
            coordinates = []

            while dataset.isOk():
                timestamp = dataset.getTimestamp()
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                if img is None:  
                    print("  -> No more images or max reached.")
                    break
                with torch.no_grad():
                    vo.track(img, depth, img_id, timestamp)
                vo_mod = vo.module if hasattr(vo, "module") else vo
                if len(vo_mod.traj3d_est) > 1:
                    x, y, z = vo_mod.traj3d_est[-1]
                    coordinates.append(f"{x} {y} {z}")
                img_id += 1

            # Save trajectory
            with open(traj_file, 'w') as file:
                file.write("\n".join(coordinates))

            runtime = time.time() - start_time
            print(f"  -> Runtime: {runtime:.2f} sec")
            with open(runtime_file, 'a') as file:
                file.write(f"{seq}: {runtime:.2f} seconds\n")
            print(f"  -> Done: {seq} with tracker {tracker_name}")

        # Clear CUDA memory after each tracker
        torch.cuda.empty_cache()
        gc.collect()

# ----- EuRoC settings -----
EUROC_PATH = "/media/shared/euroc/"
EUROC_RESULTS_DIR = "/home/ws-rtx/Documents/Projects/pyslam/results_euroc"
EUROC_CONFIG_FILE = "/home/ws-rtx/Documents/Projects/pyslam/config.yaml"
EUROC_SEQUENCES = [
    ("MH_01_easy", "settings/EuRoC_stereo.yaml"),
    ("MH_02_easy", "settings/EuRoC_stereo.yaml"),
    ("MH_03_medium", "settings/EuRoC_stereo.yaml"),
    ("MH_04_difficult", "settings/EuRoC_stereo.yaml"),
    ("MH_05_difficult", "settings/EuRoC_stereo.yaml"),
    
    ("V1_01_easy", "settings/EuRoC_stereo.yaml"),
    ("V1_02_medium", "settings/EuRoC_stereo.yaml"),
    ("V1_03_difficult", "settings/EuRoC_stereo.yaml"),

    ("V2_01_easy", "settings/EuRoC_stereo.yaml"),
    ("V2_02_medium", "settings/EuRoC_stereo.yaml"),
    ("V2_03_difficult", "settings/EuRoC_stereo.yaml"),
]

def load_groundtruth_poses_euroc(sequence_path):
    """Loads ground truth positions for a EuRoC sequence."""
    gt_file = os.path.join(sequence_path, "mav0", "state_groundtruth_estimate0", "data.csv")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    data = np.genfromtxt(gt_file, delimiter=',', skip_header=1)
    return data[:, 1:4]

def run_euroc_pipeline():
    """Runs visual odometry benchmark on the EuRoC dataset."""
    print("\n==== Running EuRoC pipeline ====")
    for tracker_name, tracker_config in TRACKER_CONFIGS.items():
        tracker_output_dir = os.path.join(EUROC_RESULTS_DIR, tracker_name)
        os.makedirs(tracker_output_dir, exist_ok=True)
        for seq, settings_file in EUROC_SEQUENCES:
            sequence_path = os.path.join(EUROC_PATH, seq)
            traj_file = os.path.join(tracker_output_dir, f"{seq}.txt")
            if os.path.exists(traj_file):
                print(f"  -> Output already exists, skipping: {tracker_name} / {seq}")
                continue

            # Update EuRoC config yaml
            with open(EUROC_CONFIG_FILE, "r") as file:
                config_yaml = yaml.safe_load(file)
            config_yaml["EUROC_DATASET"]["name"] = seq
            config_yaml["EUROC_DATASET"]["settings"] = settings_file
            config_yaml["EUROC_DATASET"]["base_path"] = EUROC_PATH
            with open(EUROC_CONFIG_FILE, "w") as file:
                yaml.dump(config_yaml, file, default_flow_style=False)

            start_time = time.time()
            # Initialize modules
            config = Config()
            dataset = dataset_factory(config)
            groundtruth = groundtruth_factory(config.dataset_settings)
            cam = PinholeCamera(config)
            num_features = config.num_features_to_extract if config.num_features_to_extract > 0 else 2000
            tracker_config['num_features'] = num_features

            feature_tracker = feature_tracker_factory(**tracker_config)
            if hasattr(feature_tracker, 'detector') and hasattr(feature_tracker.detector, 'net'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                feature_tracker.detector.net.eval()
                feature_tracker.detector.net.to(device)

            if dataset.sensor_type == SensorType.RGBD:
                vo = VisualOdometryRgbdTensor(cam, groundtruth)
            else:
                vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)

            gt_poses = load_groundtruth_poses_euroc(sequence_path)
            img_id = 0
            coordinates = []

            while dataset.isOk():
                timestamp = dataset.getTimestamp()
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                if img is None:
                    print("  -> No more images.")
                    break
                with torch.no_grad():
                    vo.track(img, depth, img_id, timestamp)
                vo_mod = vo.module if hasattr(vo, "module") else vo
                if len(vo_mod.traj3d_est) > 1:
                    x, y, z = vo_mod.traj3d_est[-1]
                    coordinates.append(f"{x} {y} {z}")
                img_id += 1

            # Save trajectory
            with open(traj_file, 'w') as file:
                file.write("\n".join(coordinates))

            runtime = time.time() - start_time
            print(f"  -> Runtime: {runtime:.2f} sec")
            print(f"  -> Done: {tracker_name} on {seq}")
            runtime_file = os.path.join(tracker_output_dir, 'runtimes.txt')
            with open(runtime_file, 'a') as file:
                file.write(f"{seq}: {runtime:.2f} seconds\n")

        # Clear CUDA memory after each tracker
        torch.cuda.empty_cache()
        gc.collect()

# ===== MAIN ENTRY POINT =====
def main():
    if DATASET == "kitti":
        run_kitti_pipeline()
    elif DATASET == "euroc":
        run_euroc_pipeline()
    elif DATASET == "both":
        run_kitti_pipeline()
        run_euroc_pipeline()
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

if __name__ == "__main__":
    main()