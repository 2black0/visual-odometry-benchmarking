# Supplementary Material for A Comparative Evaluation of Classical and Deep Learning-Based Visual Odometry Methods for Autonomous Vehicle Navigation 

This repository contains supplementary material for our comparative evaluation of classical, hybrid, and learning-based visual odometry (VO) pipelines. It includes complete configuration details, additional plots, and per-metric evaluation tables.

---
## Repository Structure

<details>
<summary><strong>Repository Structure</strong></summary>

- `results/`
  - `VO/`: Raw trajectory files (~1000 .txt)
    - `KITTI/`: KITTI sequences per method
    - `EuRoC/`: EuRoC sequences per method
  - `eval/`
    - `kitti_metrics.csv`
    - `euroc_metrics.csv`
- `trajectory_plots/`: Example plots (good/bad tracker sequences)
  - `good_seq_*.png`
  - `bad_seq_*.png`
- `source_code/`: Scripts for running and evaluating the benchmark
  - `main_vo_batch.py`: Batch runner
  - `evaluate_all.py`: Evaluation logic
  - `feature_tracker_configs.py`: All pipeline configurations
- `README.md`: Main documentation (this file)

</details>

---

## Contents
- [Benchmark Backend](#benchmark-backend)
- [Evaluation Environment](#evaluation-environment)
- [Third-party Dependencies](#third-party-dependencies)
- [Package Versions](#package-versions)
- [Evaluation Scope](#evaluation-scope)
- [Configuration List](#configuration-list)
- [KITTI Results & Ranking](#kitti-results--ranking)
- [EuRoC Results & Ranking](#euroc-results--ranking)
- [Step-by-Step Reproduction Guide](#step-by-step-reproduction-of-benchmark-results)
- [Known Isuees and Fixes](#known-issues-and-fixes) 
- [Example trajectory plots](#example-trajectory-plots)
- [References](#references)
---

## Benchmark Backend

All VO pipelines were executed using our internal evaluation framework, based on [`PySLAM`](https://github.com/luigifreda/pyslam), commit `9c20866`.

PySLAM provides a unified interface for combining feature detectors, descriptors, and matchers, as well as modules for pose estimation and trajectory export.  
It was employed to standardize pipeline execution and ensure a consistent evaluation protocol across all configurations.  
No external SLAM systems were used for evaluation logic; only the VO components specified in the configuration tables were altered.

---

## Evaluation Environment

All experiments were conducted on a local workstation with the following hardware and software setup.

### Hardware

- **CPU**: Intel Core i9-7940X  
- **GPU**: 3 × NVIDIA GeForce RTX 2080 Ti (12 GB VRAM each)  
- **RAM**: 32 GB

### Software

- **Operating System**: Kubuntu 22.04.5  
- **Python**: 3.9  
- **CUDA**: 12.6  
- **OpenCV**: 4.x
---

## Third-party Dependencies

This evaluation is based on the PySLAM framework, which integrates several third-party open-source components.  
To ensure transparency and reproducibility, the table below provides the relevant repositories and their corresponding Git commit hashes as used during our experiments.

Unless otherwise stated, all external libraries were employed without modifications. In cases where multiple versions existed, the most stable or commonly adopted implementation was selected.


<div align="center">

<table>
  <thead>
    <tr>
      <th>Thirdparty Component</th>
      <th>Hash</th>
      <th>GitHub Link</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>D2NET</td><td>0a3356c</td><td><a href="https://github.com/mihaidusmanu/d2-net">Link</a></td></tr>
    <tr><td>ORB_SLAM2</td><td>f2e6f51</td><td><a href="https://github.com/raulmur/ORB_SLAM2">Link</a></td></tr>
    <tr><td>SuperPointPretrainedNetwork</td><td>1fda796</td><td><a href="https://github.com/magicleap/SuperPointPretrainedNetwork">Link</a></td></tr>
    <tr><td>TFEAT</td><td>9434ffa</td><td><a href="https://github.com/vbalnt/tfeat">Link</a></td></tr>
    <tr><td>HARDNET</td><td>b1e9967</td><td><a href="https://github.com/DagnyT/hardnet">Link</a></td></tr>
    <tr><td>GEODESC</td><td>897a2d7</td><td><a href="https://github.com/lzx551402/geodesc">Link</a></td></tr>
    <tr><td>CONTEXTDESC</td><td>77ec9a6</td><td><a href="https://github.com/lzx551402/contextdesc">Link</a></td></tr>
    <tr><td>SOSNet Public</td><td>7ae3730</td><td><a href="https://github.com/yuruntian/SOSNet">Link</a></td></tr>
    <tr><td>L2-Net</td><td>521633c</td><td><a href="https://github.com/yuruntian/L2-Net">Link</a></td></tr>
    <tr><td>LF-NET</td><td>52abf68</td><td><a href="https://github.com/vcg-uvic/lf-net-release">Link</a></td></tr>
    <tr><td>R2D2</td><td>0ff8f6a</td><td><a href="https://github.com/naver/r2d2">Link</a></td></tr>
    <tr><td>BEBLID</td><td>b37c782</td><td><a href="https://github.com/iago-suarez/BEBLID">Link</a></td></tr>
    <tr><td>DISK</td><td>8dc6d4d</td><td><a href="https://github.com/cvlab-epfl/disk">Link</a></td></tr>
    <tr><td>XFEAT</td><td>e92685f</td><td><a href="https://github.com/verlab/accelerated_features">Link</a></td></tr>
    <tr><td>LightGlue</td><td>edb2b83</td><td><a href="https://github.com/cvg/LightGlue">Link</a></td></tr>
    <tr><td>Key.Net</td><td>24808fb</td><td><a href="https://github.com/axelBarroso/Key.Net">Link</a></td></tr>
    <tr><td>MonoVO Python (FAST, LK)</td><td>b146da3</td><td><a href="https://github.com/uoip/monoVO-python">Link</a></td></tr>
  </tbody>
</table>

</div>

---
## Package Versions

To ensure reproducibility, we provide the exact versions of all key Python packages and dependencies used during benchmarking.  
Note: Mixing library versions (e.g., incompatible FAISS or PyTorch builds) may result in runtime errors or degraded performance. We recommend strict version pinning when reproducing our results.

<div align="center">

<table>
  <thead>
    <tr>
      <th>Pip Package</th>
      <th>Version</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>opencv-python</td><td>4.10.0.84</td></tr>
    <tr><td>torch</td><td>2.2.0</td></tr>
    <tr><td>torchvision</td><td>0.17.0</td></tr>
    <tr><td>faiss-cpu</td><td>1.11.0</td></tr>
    <tr><td>faiss-gpu</td><td>1.7.2</td></tr>
    <tr><td>kornia</td><td>0.7.3</td></tr>
    <tr><td>kornia-moons</td><td>0.2.9</td></tr>
    <tr><td>tensorflow</td><td>2.19.0</td></tr>
    <tr><td>tensorflow-estimator</td><td>2.13.0</td></tr>
    <tr><td>numpy</td><td>1.26.4</td></tr>
    <tr><td>scipy</td><td>1.10.1</td></tr>
    <tr><td>scikit-learn</td><td>1.7.0</td></tr>
    <tr><td>matplotlib</td><td>3.7.5</td></tr>
  </tbody>
</table>

</div>

---
## Datasets

Two publicly available datasets were used for evaluating the VO pipelines:

- **KITTI Dataset**: Provides urban driving sequences with synchronized stereo images, ground truth poses from GPS and IMU sensors. It is widely used for evaluating vehicle-based visual odometry systems.  
  - Dataset URL: [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- **EuRoC Dataset**: This dataset is designed for micro aerial vehicle (MAV) applications and includes indoor sequences recorded with a stereo camera and ground truth obtained via a Vicon system. The dataset is suitable for evaluating VO robustness in dynamic, low-texture environments.  
  - Dataset URL: [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
    
Both datasets include annotated ground truth data for pose estimation and are crucial for evaluating the accuracy of visual odometry methods under various conditions.

---
## Evaluation Scope

A total of **52 VO configurations** were evaluated across two datasets (KITTI and EuRoC). Each configuration is defined by a detector, descriptor, and matcher combination.

The evaluation focuses on accuracy across six localization metrics:
- **ATE** – Absolute Trajectory Error
- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **MRE** – Mean Rotation Error
- **RPE** – Relative Pose Error
- **FDE** – Final Drift Error

All metrics were computed per sequence and averaged across datasets.

---

## Configuration List

### Notes on Terminology

- **BF** [1] – Brute-Force matcher (exact nearest-neighbor matching using L2 or Hamming distance)
- **FLANN** [2] – Fast Library for Approximate Nearest Neighbors (approximate, tree-based matcher)
- **LK** [3] – Lucas–Kanade optical flow tracker (dense or sparse frame-to-frame motion estimation)

Each evaluated configuration is represented as a row in the following table.  
Pipelines are grouped according to their architectural type (e.g., classical, hybrid, learning-based).




<div align="center">
  <table>
    <thead>
      <tr>
        <th>Configuration name</th>
        <th>Detector</th>
        <th>Descriptor</th>
        <th>Matcher</th>
        <th>Group</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>AKAZE</td><td>AKAZE [4]</td><td>AKAZE [4]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>ALIKED</td><td>ALIKED [5]</td><td>ALIKED [5]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>ALIKED_FLANN</td><td>ALIKED [5]</td><td>ALIKED [5]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>BRISK</td><td>BRISK [6]</td><td>BRISK [6]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>BRISK_TFEAT</td><td>BRISK [6]</td><td>TFEAT [7]</td><td>BF [1]</td><td>Hybrid</td></tr>
      <tr><td>BRISK_TFEAT_FLANN</td><td>BRISK [6]</td><td>TFEAT [7]</td><td>FLANN [2]</td><td>Hybrid</td></tr>
      <tr><td>CONTEXTDESC</td><td>CONTEXTDESC [8]</td><td>CONTEXTDESC [8]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>CONTEXTDESC_FLANN</td><td>CONTEXTDESC [8]</td><td>CONTEXTDESC [8]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>D2NET</td><td>D2NET [9]</td><td>D2NET [9]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>D2NET_FLANN</td><td>D2NET [9]</td><td>D2NET [9]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>DISK</td><td>DISK [10]</td><td>DISK [10]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>FAST_FREAK</td><td>FAST [11]</td><td>FREAK [12]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>FAST_ORB</td><td>FAST [11]</td><td>ORB [13]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>KAZE</td><td>KAZE [14]</td><td>KAZE [14]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>KEYNET</td><td>KEYNET [15]</td><td>KEYNET [15]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>KEYNET_FLANN</td><td>KEYNET [15]</td><td>KEYNET [15]</td><td>FLANN [12]</td><td>Learning-based</td></tr>
      <tr><td>KEYNETAFFNETHARDNET</td><td>KEYNETAFFNETHARDNET [16]</td><td>KEYNETAFFNETHARDNET [16]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>KEYNETAFFNETHARDNET_FLANN</td><td>KEYNETAFFNETHARDNET [16]</td><td>KEYNETAFFNETHARDNET [16]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>LFNET</td><td>LFNET [17]</td><td>LFNET [17]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>LFNET_FLANN</td><td>LFNET [17]</td><td>LFNET [17]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>LIGHTGLUE</td><td>SUPERPOINT [18]</td><td>SUPERPOINT [18]</td><td>LIGHTGLUE [19]</td><td>LightGlue-based</td></tr>
      <tr><td>LIGHTGLUE_ALIKED</td><td>ALIKED [5]</td><td>ALIKED [5]</td><td>LIGHTGLUE [19]</td><td>LightGlue-based</td></tr>
      <tr><td>LIGHTGLUE_DISK</td><td>DISK [10]</td><td>DISK [10]</td><td>LIGHTGLUE [19]</td><td>LightGlue-based</td></tr>
      <tr><td>LIGHTGLUESIFT</td><td>SIFT [20]</td><td>SIFT [20]</td><td>LIGHTGLUE  [19]</td><td>LightGlue-based</td></tr>
      <tr><td>LK_FAST</td><td>FAST [11]</td><td>None</td><td>LK [3]</td><td>Classical</td></tr>
      <tr><td>LK_SHI_TOMASI</td><td>SHI TOMASI [21]</td><td>None</td><td>LK [3]</td><td>Classical</td></tr>
      <tr><td>ORB</td><td>ORB [13]</td><td>ORB [13]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>ORB2</td><td>ORB2 [22]</td><td>ORB2 [22]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>ORB2_BEBLID</td><td>ORB2 [22]</td><td>BEBLID</td><td>BF</td><td>Hybrid</td></tr>
      <tr><td>ORB2_BEBLID_FLANN</td><td>ORB2 [22]</td><td>BEBLID</td><td>FLANN</td><td>Hybrid</td></tr>
      <tr><td>ORB2_FREAK</td><td>ORB2 [22]</td><td>FREAK [12]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>ORB2_FREAK_FLANN</td><td>ORB2 [22]</td><td>FREAK [12]</td><td>FLANN [2]</td><td>Classical</td></tr>
      <tr><td>ORB2_HARDNET</td><td>ORB2 [22]</td><td>HARDNET [23]</td><td>BF [1]</td><td>Hybrid</td></tr>
      <tr><td>ORB2_HARDNET_FLANN</td><td>ORB2 [22]</td><td>HARDNET [23]</td><td>FLANN [2]</td><td>Hybrid</td></tr>
      <tr><td>ORB2_L2NET</td><td>ORB2 [22]</td><td>L2NET [24]</td><td>BF [1]</td><td>Hybrid</td></tr>
      <tr><td>ORB2_L2NET_FLANN</td><td>ORB [22]</td><td>L2NET [24]</td><td>FLANN [2]</td><td>Hybrid</td></tr>
      <tr><td>ORB2_SOSNET</td><td>ORB2 [22]</td><td>SOSNET [25]</td><td>BF [1]</td><td>Hybrid</td></tr>
      <tr><td>ORB2_SOSNET_FLANN</td><td>ORB2 [22]</td><td>SOSNET [25]</td><td>FLANN [2]</td><td>Hybrid</td></tr>
      <tr><td>R2D2</td><td>R2D2 [26]</td><td>R2D2 [26]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>R2D2_FLANN</td><td>R2D2 [26]</td><td>R2D2 [26]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>ROOT_SIFT</td><td>ROOT_SIFT [27]</td><td>ROOT_SIFT [27]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>ROOT_SIFT_FLANN</td><td>ROOT_SIFT [27]</td><td>ROOT_SIFT [27]</td><td>FLANN [2]</td><td>Classical</td></tr>
      <tr><td>SHI_TOMASI_FREAK</td><td>SHI TOMASI [21]</td><td>FREAK [12]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>SHI_TOMASI_ORB</td><td>SHI TOMASI [21]</td><td>ORB [13]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>SIFT</td><td>SIFT [20]</td><td>SIFT [20]</td><td>BF [1]</td><td>Classical</td></tr>
      <tr><td>SIFT_FLANN</td><td>SIFT [20]</td><td>SIFT [20]</td><td>FLANN [2]</td><td>Classical</td></tr>
      <tr><td>SUPERPOINT</td><td>SUPERPOINT [18]</td><td>SUPERPOINT [18]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>SUPERPOINT_FLANN</td><td>SUPERPOINT [18]</td><td>SUPERPOINT [18]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>XFEAT</td><td>XFEAT [28]</td><td>XFEAT [28]</td><td>BF [1]</td><td>Learning-based</td></tr>
      <tr><td>XFEAT_FLANN</td><td>XFEAT [28]</td><td>XFEAT [28]</td><td>FLANN [2]</td><td>Learning-based</td></tr>
      <tr><td>XFEAT_LIGHTGLUE</td><td>XFEAT [28]</td><td>XFEAT [28]</td><td>LIGHTGLUE [19]</td><td>LightGlue-based</td></tr>
      <tr><td>XFEAT_XFEAT</td><td>XFEAT [28]</td><td>XFEAT [28]</td><td>XFEAT [28]</td><td>Learning-based</td></tr>
    </tbody>
  </table>
</div>






### Notes on Pipeline Configuration

The VO pipelines tested in this benchmark were constructed using combinations of classical and learned detectors, descriptors, and matchers.  
Default parameters were retained for most third-party methods, with the following adjustments applied to ensure consistent evaluation:


- **Matcher thresholding** was standardized across pipelines to remove bias from aggressive filtering.
- **Descriptor normalization** was applied where needed (e.g., ROOT-SIFT, SOSNet) to unify feature spaces.
- **SLAM backends** (e.g., ORB-SLAM2) were minimally wrapped to extract trajectory outputs for metrics computation.

The full list of evaluated pipelines, along with their components and group assignments, is provided in the configuration table above.

---

## KITTI Results & Ranking


The table below lists all evaluated visual odometry pipelines on the KITTI dataset.  
For each configuration, we report six localization metrics:

- **ATE** – Absolute Trajectory Error  
- **MAE** – Mean Absolute Error  
- **MSE** – Mean Squared Error  
- **MRE** – Mean Rotation Error  
- **RPE** – Relative Pose Error  
- **FDE** – Final Drift Error

After computing the raw metric values, we independently ranked all pipelines **per metric**, from best to worst.  
These per-metric ranks were then averaged to obtain a final **average rank** for each configuration.

The table is sorted by this average rank in ascending order, with the best overall configurations at the top.

<table  cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th rowspan="2">Tracker</th>
      <th colspan="6">Results</th>
      <th colspan="7">Ranks</th>
    </tr>
    <tr>
      <th>ATE [m]</th>
      <th>MAE [m]</th>
      <th>MSE [m²]</th>
      <th>MRE</th>
      <th>RPE [m]</th>
      <th>FDE [m]</th>
      <th>ATE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MRE</th>
      <th>RPE</th>
      <th>FDE</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td>LIGHTGLUESIFT</td>
    <td>13.0110</td>
    <td>11.6100</td>
    <td>331.870</td>
    <td>0.0798</td>
    <td>0.0475</td>
    <td>18.8950</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1.00</td>
  </tr>
  <tr>
    <td>LIGHTGLUE_ALIKED</td>
    <td>13.9260</td>
    <td>12.2190</td>
    <td>566.910</td>
    <td>0.1003</td>
    <td>0.0534</td>
    <td>20.9820</td>
    <td>2</td>
    <td>2</td>
    <td>3</td>
    <td>6</td>
    <td>2</td>
    <td>2</td>
    <td>2.83</td>
  </tr>
  <tr>
    <td>ROOT_SIFT</td>
    <td>14.1850</td>
    <td>12.2280</td>
    <td>480.310</td>
    <td>0.1249</td>
    <td>0.0525</td>
    <td>20.5350</td>
    <td>3</td>
    <td>3</td>
    <td>2</td>
    <td>13</td>
    <td>3</td>
    <td>3</td>
    <td>4.50</td>
  </tr>
  <tr>
    <td>ROOT_SIFT_FLANN</td>
    <td>14.1850</td>
    <td>12.2280</td>
    <td>480.310</td>
    <td>0.1249</td>
    <td>0.0525</td>
    <td>20.5350</td>
    <td>3</td>
    <td>3</td>
    <td>2</td>
    <td>13</td>
    <td>3</td>
    <td>3</td>
    <td>4.50</td>
  </tr>
  <tr>
    <td>SIFT</td>
    <td>16.2120</td>
    <td>14.2010</td>
    <td>652.010</td>
    <td>0.1093</td>
    <td>0.0558</td>
    <td>22.7010</td>
    <td>4</td>
    <td>4</td>
    <td>4</td>
    <td>9</td>
    <td>4</td>
    <td>4</td>
    <td>4.83</td>
  </tr>
  <tr>
    <td>SIFT_FLANN</td>
    <td>16.2120</td>
    <td>14.2010</td>
    <td>652.010</td>
    <td>0.1093</td>
    <td>0.0558</td>
    <td>22.7010</td>
    <td>4</td>
    <td>4</td>
    <td>4</td>
    <td>9</td>
    <td>4</td>
    <td>4</td>
    <td>4.83</td>
  </tr>
  <tr>
    <td>CONTEXTDESC</td>
    <td>16.7580</td>
    <td>14.4440</td>
    <td>911.140</td>
    <td>0.0981</td>
    <td>0.0566</td>
    <td>23.2960</td>
    <td>5</td>
    <td>5</td>
    <td>7</td>
    <td>5</td>
    <td>5</td>
    <td>5</td>
    <td>5.33</td>
  </tr>
  <tr>
    <td>CONTEXTDESC_FLANN</td>
    <td>16.7580</td>
    <td>14.4440</td>
    <td>911.140</td>
    <td>0.0981</td>
    <td>0.0566</td>
    <td>23.2960</td>
    <td>5</td>
    <td>5</td>
    <td>7</td>
    <td>5</td>
    <td>5</td>
    <td>5</td>
    <td>5.33</td>
  </tr>
  <tr>
    <td>LIGHTGLUE</td>
    <td>18.0660</td>
    <td>15.6630</td>
    <td>786.310</td>
    <td>0.1309</td>
    <td>0.0627</td>
    <td>25.1770</td>
    <td>6</td>
    <td>6</td>
    <td>6</td>
    <td>15</td>
    <td>6</td>
    <td>6</td>
    <td>7.50</td>
  </tr>
  <tr>
    <td>BRISK_TFEAT</td>
    <td>20.3070</td>
    <td>17.1990</td>
    <td>1260.17</td>
    <td>0.1068</td>
    <td>0.0701</td>
    <td>30.4660</td>
    <td>7</td>
    <td>7</td>
    <td>8</td>
    <td>8</td>
    <td>8</td>
    <td>8</td>
    <td>7.67</td>
  </tr>
  <tr>
    <td>BRISK_TFEAT_FLANN</td>
    <td>20.3070</td>
    <td>17.1990</td>
    <td>1260.17</td>
    <td>0.1068</td>
    <td>0.0701</td>
    <td>30.4660</td>
    <td>7</td>
    <td>7</td>
    <td>8</td>
    <td>8</td>
    <td>8</td>
    <td>8</td>
    <td>7.67</td>
  </tr>
  <tr>
    <td>ALIKED</td>
    <td>22.2190</td>
    <td>19.0710</td>
    <td>2019.21</td>
    <td>0.0949</td>
    <td>0.0699</td>
    <td>32.5520</td>
    <td>10</td>
    <td>10</td>
    <td>11</td>
    <td>4</td>
    <td>7</td>
    <td>7</td>
    <td>8.17</td>
  </tr>
  <tr>
    <td>ALIKED_FLANN</td>
    <td>22.2190</td>
    <td>19.0710</td>
    <td>2019.21</td>
    <td>0.0949</td>
    <td>0.0699</td>
    <td>32.5520</td>
    <td>10</td>
    <td>10</td>
    <td>11</td>
    <td>4</td>
    <td>7</td>
    <td>7</td>
    <td>8.17</td>
  </tr>
  <tr>
    <td>LK_SHI_TOMASI</td>
    <td>20.8740</td>
    <td>16.6310</td>
    <td>1456.71</td>
    <td>0.0815</td>
    <td>0.0794</td>
    <td>36.6290</td>
    <td>8</td>
    <td>8</td>
    <td>9</td>
    <td>2</td>
    <td>12</td>
    <td>12</td>
    <td>8.50</td>
  </tr>
  <tr>
    <td>BRISK</td>
    <td>23.7110</td>
    <td>19.8490</td>
    <td>1952.25</td>
    <td>0.1062</td>
    <td>0.0787</td>
    <td>32.7910</td>
    <td>12</td>
    <td>12</td>
    <td>10</td>
    <td>7</td>
    <td>11</td>
    <td>11</td>
    <td>10.50</td>
  </tr>
  <tr>
    <td>LK_FAST</td>
    <td>22.4240</td>
    <td>18.3670</td>
    <td>2097.37</td>
    <td>0.0880</td>
    <td>0.0798</td>
    <td>33.6760</td>
    <td>11</td>
    <td>11</td>
    <td>12</td>
    <td>3</td>
    <td>13</td>
    <td>13</td>
    <td>10.50</td>
  </tr>
  <tr>
    <td>LIGHTGLUE_DISK</td>
    <td>21.3700</td>
    <td>17.3790</td>
    <td>775.910</td>
    <td>0.1638</td>
    <td>0.0729</td>
    <td>29.5250</td>
    <td>9</td>
    <td>9</td>
    <td>5</td>
    <td>25</td>
    <td>9</td>
    <td>9</td>
    <td>11.00</td>
  </tr>
  <tr>
    <td>SUPERPOINT_FLANN</td>
    <td>24.1970</td>
    <td>20.2430</td>
    <td>2613.87</td>
    <td>0.1221</td>
    <td>0.0744</td>
    <td>32.0100</td>
    <td>14</td>
    <td>14</td>
    <td>14</td>
    <td>12</td>
    <td>10</td>
    <td>10</td>
    <td>12.33</td>
  </tr>
  <tr>
    <td>KAZE</td>
    <td>24.0820</td>
    <td>19.8830</td>
    <td>2442.81</td>
    <td>0.1116</td>
    <td>0.0800</td>
    <td>33.4730</td>
    <td>13</td>
    <td>13</td>
    <td>13</td>
    <td>10</td>
    <td>14</td>
    <td>14</td>
    <td>12.83</td>
  </tr>
  <tr>
    <td>SUPERPOINT</td>
    <td>26.2090</td>
    <td>21.9270</td>
    <td>3192.34</td>
    <td>0.1373</td>
    <td>0.0826</td>
    <td>37.6180</td>
    <td>15</td>
    <td>15</td>
    <td>17</td>
    <td>18</td>
    <td>15</td>
    <td>15</td>
    <td>15.83</td>
  </tr>
  <tr>
    <td>AKAZE</td>
    <td>27.6970</td>
    <td>23.1340</td>
    <td>2782.72</td>
    <td>0.1555</td>
    <td>0.0847</td>
    <td>40.5020</td>
    <td>16</td>
    <td>16</td>
    <td>15</td>
    <td>19</td>
    <td>16</td>
    <td>16</td>
    <td>16.33</td>
  </tr>
  <tr>
    <td>LFNET_FLANN</td>
    <td>29.8500</td>
    <td>24.5780</td>
    <td>3615.06</td>
    <td>0.1340</td>
    <td>0.0989</td>
    <td>42.4160</td>
    <td>17</td>
    <td>17</td>
    <td>18</td>
    <td>16</td>
    <td>17</td>
    <td>17</td>
    <td>17.00</td>
  </tr>
  <tr>
    <td>KEYNETAFFNETHARDNET</td>
    <td>30.7840</td>
    <td>24.6840</td>
    <td>4771.05</td>
    <td>0.1635</td>
    <td>0.0998</td>
    <td>48.4960</td>
    <td>18</td>
    <td>18</td>
    <td>21</td>
    <td>23</td>
    <td>18</td>
    <td>18</td>
    <td>19.33</td>
  </tr>
  <tr>
    <td>DISK</td>
    <td>32.1130</td>
    <td>26.1250</td>
    <td>4464.43</td>
    <td>0.1556</td>
    <td>0.1002</td>
    <td>51.5280</td>
    <td>20</td>
    <td>20</td>
    <td>20</td>
    <td>20</td>
    <td>19</td>
    <td>20</td>
    <td>19.83</td>
  </tr>
  <tr>
    <td>KEYNETAFFNETHARDNET_FLANN</td>
    <td>30.7910</td>
    <td>24.6870</td>
    <td>4771.26</td>
    <td>0.1636</td>
    <td>0.0998</td>
    <td>48.5260</td>
    <td>19</td>
    <td>19</td>
    <td>22</td>
    <td>24</td>
    <td>18</td>
    <td>19</td>
    <td>20.17</td>
  </tr>
  <tr>
    <td>KEYNET</td>
    <td>34.5380</td>
    <td>28.3620</td>
    <td>6264.23</td>
    <td>0.1562</td>
    <td>0.1015</td>
    <td>48.0140</td>
    <td>21</td>
    <td>21</td>
    <td>24</td>
    <td>21</td>
    <td>20</td>
    <td>21</td>
    <td>21.33</td>
  </tr>
  <tr>
    <td>FAST_ORB</td>
    <td>34.8040</td>
    <td>28.4860</td>
    <td>5140.04</td>
    <td>0.1203</td>
    <td>0.1289</td>
    <td>56.2510</td>
    <td>23</td>
    <td>23</td>
    <td>23</td>
    <td>11</td>
    <td>24</td>
    <td>25</td>
    <td>21.50</td>
  </tr>
  <tr>
    <td>KEYNET_FLANN</td>
    <td>34.5550</td>
    <td>28.3750</td>
    <td>6264.69</td>
    <td>0.1564</td>
    <td>0.1016</td>
    <td>48.0660</td>
    <td>22</td>
    <td>22</td>
    <td>25</td>
    <td>22</td>
    <td>21</td>
    <td>22</td>
    <td>22.33</td>
  </tr>
  <tr>
    <td>XFEAT_XFEAT</td>
    <td>35.0820</td>
    <td>29.0060</td>
    <td>5009.08</td>
    <td>0.1674</td>
    <td>0.1119</td>
    <td>50.2580</td>
    <td>24</td>
    <td>24</td>
    <td>16</td>
    <td>26</td>
    <td>22</td>
    <td>23</td>
    <td>22.50</td>
  </tr>
  <tr>
    <td>ORB</td>
    <td>40.3980</td>
    <td>32.7650</td>
    <td>8422.45</td>
    <td>0.1253</td>
    <td>0.1344</td>
    <td>62.0220</td>
    <td>26</td>
    <td>26</td>
    <td>26</td>
    <td>14</td>
    <td>26</td>
    <td>27</td>
    <td>24.17</td>
  </tr>
  <tr>
    <td>FAST_FREAK</td>
    <td>39.2780</td>
    <td>31.6920</td>
    <td>4391.08</td>
    <td>0.1919</td>
    <td>0.1291</td>
    <td>59.3970</td>
    <td>25</td>
    <td>25</td>
    <td>19</td>
    <td>31</td>
    <td>25</td>
    <td>26</td>
    <td>25.17</td>
  </tr>
  <tr>
    <td>XFEAT_LIGHTGLUE</td>
    <td>48.6110</td>
    <td>40.1410</td>
    <td>11228.58</td>
    <td>0.1772</td>
    <td>0.1287</td>
    <td>70.0770</td>
    <td>28</td>
    <td>28</td>
    <td>28</td>
    <td>28</td>
    <td>23</td>
    <td>24</td>
    <td>26.50</td>
  </tr>
  <tr>
    <td>LFNET</td>
    <td>53.0400</td>
    <td>43.2680</td>
    <td>17688.53</td>
    <td>0.1352</td>
    <td>0.1487</td>
    <td>80.6940</td>
    <td>29</td>
    <td>29</td>
    <td>29</td>
    <td>17</td>
    <td>28</td>
    <td>29</td>
    <td>26.83</td>
  </tr>
  <tr>
    <td>XFEAT</td>
    <td>46.3220</td>
    <td>38.1870</td>
    <td>9050.90</td>
    <td>0.1698</td>
    <td>0.1413</td>
    <td>70.3990</td>
    <td>27</td>
    <td>27</td>
    <td>27</td>
    <td>27</td>
    <td>27</td>
    <td>28</td>
    <td>27.17</td>
  </tr>
  <tr>
    <td>XFEAT_FLANN</td>
    <td>61.9720</td>
    <td>50.1340</td>
    <td>18998.03</td>
    <td>0.1805</td>
    <td>0.1832</td>
    <td>87.4720</td>
    <td>30</td>
    <td>30</td>
    <td>32</td>
    <td>30</td>
    <td>29</td>
    <td>30</td>
    <td>30.17</td>
  </tr>
  <tr>
    <td>R2D2</td>
    <td>75.1340</td>
    <td>61.2400</td>
    <td>25226.36</td>
    <td>0.3380</td>
    <td>0.2159</td>
    <td>106.516</td>
    <td>32</td>
    <td>32</td>
    <td>33</td>
    <td>33</td>
    <td>30</td>
    <td>31</td>
    <td>31.83</td>
  </tr>
  <tr>
    <td>R2D2_FLANN</td>
    <td>75.1340</td>
    <td>61.2400</td>
    <td>25226.36</td>
    <td>0.3380</td>
    <td>0.2159</td>
    <td>106.516</td>
    <td>32</td>
    <td>32</td>
    <td>33</td>
    <td>33</td>
    <td>30</td>
    <td>31</td>
    <td>31.83</td>
  </tr>
  <tr>
    <td>ORB2_HARDNET_FLANN</td>
    <td>74.0320</td>
    <td>59.1200</td>
    <td>18460.27</td>
    <td>0.4830</td>
    <td>0.2435</td>
    <td>117.619</td>
    <td>31</td>
    <td>31</td>
    <td>31</td>
    <td>38</td>
    <td>31</td>
    <td>32</td>
    <td>32.33</td>
  </tr>
  <tr>
    <td>SHI_TOMASI_ORB</td>
    <td>92.0010</td>
    <td>74.2420</td>
    <td>18204.23</td>
    <td>0.5331</td>
    <td>0.2792</td>
    <td>125.757</td>
    <td>34</td>
    <td>34</td>
    <td>30</td>
    <td>39</td>
    <td>34</td>
    <td>35</td>
    <td>34.33</td>
  </tr>
  <tr>
    <td>ORB2_L2NET_FLANN</td>
    <td>93.3450</td>
    <td>76.4970</td>
    <td>45504.04</td>
    <td>0.3921</td>
    <td>0.2622</td>
    <td>131.875</td>
    <td>35</td>
    <td>35</td>
    <td>37</td>
    <td>34</td>
    <td>33</td>
    <td>34</td>
    <td>34.67</td>
  </tr>
  <tr>
    <td>ORB2_HARDNET</td>
    <td>88.9610</td>
    <td>71.2830</td>
    <td>32638.42</td>
    <td>0.4645</td>
    <td>0.2796</td>
    <td>127.001</td>
    <td>33</td>
    <td>33</td>
    <td>36</td>
    <td>37</td>
    <td>35</td>
    <td>36</td>
    <td>35.00</td>
  </tr>
  <tr>
    <td>D2NET</td>
    <td>102.615</td>
    <td>84.5970</td>
    <td>52626.73</td>
    <td>0.1803</td>
    <td>0.2523</td>
    <td>155.260</td>
    <td>39</td>
    <td>39</td>
    <td>39</td>
    <td>29</td>
    <td>32</td>
    <td>33</td>
    <td>35.17</td>
  </tr>
  <tr>
    <td>D2NET_FLANN</td>
    <td>102.615</td>
    <td>84.5970</td>
    <td>52626.73</td>
    <td>0.1803</td>
    <td>0.2523</td>
    <td>155.260</td>
    <td>39</td>
    <td>39</td>
    <td>39</td>
    <td>29</td>
    <td>32</td>
    <td>33</td>
    <td>35.17</td>
  </tr>
  <tr>
    <td>ORB2_FREAK_FLANN</td>
    <td>101.025</td>
    <td>83.4080</td>
    <td>26309.12</td>
    <td>0.3185</td>
    <td>0.2822</td>
    <td>137.462</td>
    <td>38</td>
    <td>38</td>
    <td>34</td>
    <td>32</td>
    <td>37</td>
    <td>38</td>
    <td>36.17</td>
  </tr>
  <tr>
    <td>ORB2_L2NET</td>
    <td>99.2340</td>
    <td>81.8250</td>
    <td>53364.34</td>
    <td>0.4451</td>
    <td>0.2805</td>
    <td>140.787</td>
    <td>37</td>
    <td>37</td>
    <td>38</td>
    <td>35</td>
    <td>36</td>
    <td>37</td>
    <td>36.67</td>
  </tr>
  <tr>
    <td>ORB2</td>
    <td>95.9720</td>
    <td>78.6860</td>
    <td>30526.45</td>
    <td>0.5258</td>
    <td>0.3194</td>
    <td>119.757</td>
    <td>36</td>
    <td>36</td>
    <td>35</td>
    <td>40</td>
    <td>38</td>
    <td>39</td>
    <td>37.33</td>
  </tr>
  <tr>
    <td>ORB2_FREAK</td>
    <td>148.337</td>
    <td>121.136</td>
    <td>72402.73</td>
    <td>0.4485</td>
    <td>0.3971</td>
    <td>210.862</td>
    <td>42</td>
    <td>42</td>
    <td>41</td>
    <td>36</td>
    <td>39</td>
    <td>40</td>
    <td>40.00</td>
  </tr>
  <tr>
    <td>ORB2_BEBLID</td>
    <td>144.421</td>
    <td>120.702</td>
    <td>79481.60</td>
    <td>0.6561</td>
    <td>0.4150</td>
    <td>178.970</td>
    <td>41</td>
    <td>41</td>
    <td>42</td>
    <td>41</td>
    <td>40</td>
    <td>41</td>
    <td>41.00</td>
  </tr>
  <tr>
    <td>ORB2_SOSNET_FLANN</td>
    <td>143.941</td>
    <td>118.047</td>
    <td>72016.75</td>
    <td>0.7061</td>
    <td>0.4198</td>
    <td>185.069</td>
    <td>40</td>
    <td>40</td>
    <td>40</td>
    <td>43</td>
    <td>41</td>
    <td>42</td>
    <td>41.00</td>
  </tr>
  <tr>
    <td>ORB2_BEBLID_FLANN</td>
    <td>148.642</td>
    <td>124.166</td>
    <td>89389.80</td>
    <td>0.6915</td>
    <td>0.4288</td>
    <td>195.242</td>
    <td>43</td>
    <td>43</td>
    <td>44</td>
    <td>42</td>
    <td>42</td>
    <td>43</td>
    <td>42.83</td>
  </tr>
  <tr>
    <td>ORB2_SOSNET</td>
    <td>156.212</td>
    <td>129.310</td>
    <td>84983.08</td>
    <td>0.7357</td>
    <td>0.4423</td>
    <td>213.768</td>
    <td>44</td>
    <td>44</td>
    <td>43</td>
    <td>44</td>
    <td>43</td>
    <td>44</td>
    <td>43.67</td>
  </tr>
  <tr>
    <td>SHI_TOMASI_FREAK</td>
    <td>323.900</td>
    <td>276.163</td>
    <td>137510.73</td>
    <td>1.4315</td>
    <td>0.9315</td>
    <td>398.667</td>
    <td>45</td>
    <td>45</td>
    <td>45</td>
    <td>45</td>
    <td>44</td>
    <td>45</td>
    <td>44.83</td>
  </tr>
</tbody>
</table>



---

## EuRoC Results & Ranking

This table contains the complete evaluation results for all VO pipelines tested on the EuRoC dataset.  
We report the same six metrics used for KITTI:

- **ATE** – Absolute Trajectory Error  
- **MAE** – Mean Absolute Error  
- **MSE** – Mean Squared Error  
- **MRE** – Mean Rotation Error  
- **RPE** – Relative Pose Error  
- **FDE** – Final Drift Error

For each metric, pipelines were ranked independently based on their raw values.  
We then calculated the **average of these six ranks** to obtain a single overall score per configuration.

The table is sorted by average rank, with lower values indicating better overall performance.

<table  cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th rowspan="2">Tracker</th>
      <th colspan="6">Results</th>
      <th colspan="7">Ranks</th>
    </tr>
    <tr>
      <th>ATE [m]</th><th>MAE [m]</th><th>MSE [m²]</th><th>MRE</th><th>RPE [m]</th><th>FDE [m]</th>
      <th>ATE</th><th>MAE</th><th>MSE</th><th>MRE</th><th>RPE</th><th>FDE</th><th>Average</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td>LIGHTGLUESIFT</td>
    <td>1.5500</td>
    <td>1.3980</td>
    <td>2.6530</td>
    <td>0.388</td>
    <td>0.0190</td>
    <td>2.1790</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>2</td>
    <td>1.17</td>
  </tr>
  <tr>
    <td>D2NET</td>
    <td>1.7290</td>
    <td>1.5880</td>
    <td>3.1200</td>
    <td>0.436</td>
    <td>0.0250</td>
    <td>2.2240</td>
    <td>2</td>
    <td>2</td>
    <td>2</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>2.50</td>
  </tr>
  <tr>
    <td>D2NET_FLANN</td>
    <td>1.7290</td>
    <td>1.5880</td>
    <td>3.1200</td>
    <td>0.436</td>
    <td>0.0250</td>
    <td>2.2240</td>
    <td>2</td>
    <td>2</td>
    <td>2</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>2.50</td>
  </tr>
  <tr>
    <td>KEYNET</td>
    <td>2.0220</td>
    <td>1.8510</td>
    <td>4.6750</td>
    <td>0.459</td>
    <td>0.0250</td>
    <td>2.4050</td>
    <td>4</td>
    <td>4</td>
    <td>4</td>
    <td>4</td>
    <td>3</td>
    <td>5</td>
    <td>4.00</td>
  </tr>
  <tr>
    <td>KEYNET_FLANN</td>
    <td>2.0220</td>
    <td>1.8510</td>
    <td>4.6750</td>
    <td>0.459</td>
    <td>0.0250</td>
    <td>2.4050</td>
    <td>4</td>
    <td>4</td>
    <td>4</td>
    <td>4</td>
    <td>3</td>
    <td>5</td>
    <td>4.00</td>
  </tr>
  <tr>
    <td>KAZE</td>
    <td>2.1090</td>
    <td>1.9140</td>
    <td>5.0950</td>
    <td>0.478</td>
    <td>0.0240</td>
    <td>2.2950</td>
    <td>7</td>
    <td>8</td>
    <td>7</td>
    <td>7</td>
    <td>2</td>
    <td>4</td>
    <td>5.83</td>
  </tr>
  <tr>
    <td>LFNET_FLANN</td>
    <td>2.0890</td>
    <td>1.9070</td>
    <td>4.9020</td>
    <td>0.486</td>
    <td>0.0250</td>
    <td>2.5380</td>
    <td>6</td>
    <td>6</td>
    <td>5</td>
    <td>8</td>
    <td>3</td>
    <td>7</td>
    <td>5.83</td>
  </tr>
  <tr>
    <td>LFNET</td>
    <td>2.0100</td>
    <td>1.8260</td>
    <td>4.3390</td>
    <td>0.477</td>
    <td>0.0260</td>
    <td>2.6790</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>6</td>
    <td>4</td>
    <td>16</td>
    <td>5.83</td>
  </tr>
  <tr>
    <td>KEYNETAFFNETHARDNET</td>
    <td>2.0860</td>
    <td>1.8920</td>
    <td>5.0000</td>
    <td>0.468</td>
    <td>0.0250</td>
    <td>2.6680</td>
    <td>5</td>
    <td>5</td>
    <td>6</td>
    <td>5</td>
    <td>3</td>
    <td>15</td>
    <td>6.50</td>
  </tr>
  <tr>
    <td>KEYNETAFFNETHARDNET_FLANN</td>
    <td>2.0860</td>
    <td>1.8920</td>
    <td>5.0000</td>
    <td>0.468</td>
    <td>0.0250</td>
    <td>2.6680</td>
    <td>5</td>
    <td>5</td>
    <td>6</td>
    <td>5</td>
    <td>3</td>
    <td>15</td>
    <td>6.50</td>
  </tr>
  <tr>
    <td>ORB</td>
    <td>2.1310</td>
    <td>1.9090</td>
    <td>4.6750</td>
    <td>0.521</td>
    <td>0.0270</td>
    <td>2.1280</td>
    <td>8</td>
    <td>7</td>
    <td>4</td>
    <td>26</td>
    <td>5</td>
    <td>1</td>
    <td>8.50</td>
  </tr>
  <tr>
    <td>LIGHTGLUE_DISK</td>
    <td>2.1820</td>
    <td>1.9770</td>
    <td>5.3710</td>
    <td>0.498</td>
    <td>0.0260</td>
    <td>2.6240</td>
    <td>9</td>
    <td>10</td>
    <td>10</td>
    <td>10</td>
    <td>4</td>
    <td>13</td>
    <td>9.33</td>
  </tr>
  <tr>
    <td>LIGHTGLUE</td>
    <td>2.2520</td>
    <td>2.0670</td>
    <td>5.5070</td>
    <td>0.512</td>
    <td>0.0260</td>
    <td>2.5800</td>
    <td>13</td>
    <td>14</td>
    <td>13</td>
    <td>13</td>
    <td>4</td>
    <td>9</td>
    <td>11.00</td>
  </tr>
  <tr>
    <td>SUPERPOINT</td>
    <td>2.2410</td>
    <td>2.0450</td>
    <td>5.3150</td>
    <td>0.516</td>
    <td>0.0270</td>
    <td>2.8630</td>
    <td>12</td>
    <td>11</td>
    <td>9</td>
    <td>14</td>
    <td>5</td>
    <td>24</td>
    <td>12.50</td>
  </tr>
  <tr>
    <td>SUPERPOINT_FLANN</td>
    <td>2.2410</td>
    <td>2.0450</td>
    <td>5.3150</td>
    <td>0.516</td>
    <td>0.0270</td>
    <td>2.8630</td>
    <td>12</td>
    <td>11</td>
    <td>9</td>
    <td>14</td>
    <td>5</td>
    <td>24</td>
    <td>12.50</td>
  </tr>
  <tr>
    <td>ORB2_FREAK</td>
    <td>2.2400</td>
    <td>2.0490</td>
    <td>5.4220</td>
    <td>0.51</td>
    <td>0.0330</td>
    <td>2.9180</td>
    <td>11</td>
    <td>12</td>
    <td>11</td>
    <td>11</td>
    <td>10</td>
    <td>26</td>
    <td>13.50</td>
  </tr>
  <tr>
    <td>ORB2_FREAK_FLANN</td>
    <td>2.1900</td>
    <td>1.9690</td>
    <td>5.1460</td>
    <td>0.496</td>
    <td>0.0320</td>
    <td>3.2720</td>
    <td>10</td>
    <td>9</td>
    <td>8</td>
    <td>9</td>
    <td>9</td>
    <td>39</td>
    <td>14.00</td>
  </tr>
  <tr>
    <td>SHI_TOMASI_ORB</td>
    <td>2.2410</td>
    <td>2.0630</td>
    <td>5.7080</td>
    <td>0.511</td>
    <td>0.0270</td>
    <td>2.9800</td>
    <td>12</td>
    <td>13</td>
    <td>14</td>
    <td>12</td>
    <td>5</td>
    <td>29</td>
    <td>14.17</td>
  </tr>
  <tr>
    <td>BRISK</td>
    <td>2.3890</td>
    <td>2.1780</td>
    <td>6.0730</td>
    <td>0.52</td>
    <td>0.0260</td>
    <td>2.6220</td>
    <td>20</td>
    <td>20</td>
    <td>16</td>
    <td>16</td>
    <td>4</td>
    <td>12</td>
    <td>14.67</td>
  </tr>
  <tr>
    <td>LIGHTGLUE_ALIKED</td>
    <td>2.3390</td>
    <td>2.1270</td>
    <td>5.9310</td>
    <td>0.531</td>
    <td>0.0270</td>
    <td>2.8140</td>
    <td>15</td>
    <td>16</td>
    <td>15</td>
    <td>19</td>
    <td>5</td>
    <td>19</td>
    <td>14.83</td>
  </tr>
  <tr>
    <td>SHI_TOMASI_FREAK</td>
    <td>2.2580</td>
    <td>2.0680</td>
    <td>5.4550</td>
    <td>0.534</td>
    <td>0.0370</td>
    <td>2.7370</td>
    <td>14</td>
    <td>15</td>
    <td>12</td>
    <td>21</td>
    <td>11</td>
    <td>17</td>
    <td>15.00</td>
  </tr>
  <tr>
    <td>LK_SHI_TOMASI</td>
    <td>2.3420</td>
    <td>2.1520</td>
    <td>6.2590</td>
    <td>0.537</td>
    <td>0.0280</td>
    <td>2.6080</td>
    <td>16</td>
    <td>18</td>
    <td>20</td>
    <td>23</td>
    <td>6</td>
    <td>11</td>
    <td>15.67</td>
  </tr>
  <tr>
    <td>R2D2</td>
    <td>2.3480</td>
    <td>2.1490</td>
    <td>6.1040</td>
    <td>0.524</td>
    <td>0.0280</td>
    <td>2.8320</td>
    <td>17</td>
    <td>17</td>
    <td>17</td>
    <td>17</td>
    <td>6</td>
    <td>22</td>
    <td>16.00</td>
  </tr>
  <tr>
    <td>R2D2_FLANN</td>
    <td>2.3480</td>
    <td>2.1490</td>
    <td>6.1040</td>
    <td>0.524</td>
    <td>0.0280</td>
    <td>2.8320</td>
    <td>17</td>
    <td>17</td>
    <td>17</td>
    <td>17</td>
    <td>6</td>
    <td>22</td>
    <td>16.00</td>
  </tr>
  <tr>
    <td>SIFT</td>
    <td>2.3610</td>
    <td>2.1550</td>
    <td>6.2230</td>
    <td>0.524</td>
    <td>0.0500</td>
    <td>2.6500</td>
    <td>18</td>
    <td>19</td>
    <td>19</td>
    <td>29</td>
    <td>8</td>
    <td>14</td>
    <td>17.83</td>
  </tr>
  <tr>
    <td>SIFT_FLANN</td>
    <td>2.3610</td>
    <td>2.1550</td>
    <td>6.2230</td>
    <td>0.524</td>
    <td>0.0500</td>
    <td>2.6500</td>
    <td>18</td>
    <td>19</td>
    <td>19</td>
    <td>29</td>
    <td>8</td>
    <td>14</td>
    <td>17.83</td>
  </tr>
  <tr>
    <td>ORB2_L2NET</td>
    <td>2.3880</td>
    <td>2.1950</td>
    <td>6.4810</td>
    <td>0.518</td>
    <td>0.0280</td>
    <td>2.9680</td>
    <td>19</td>
    <td>21</td>
    <td>22</td>
    <td>15</td>
    <td>6</td>
    <td>28</td>
    <td>18.50</td>
  </tr>
  <tr>
    <td>FAST_ORB</td>
    <td>2.4220</td>
    <td>2.2320</td>
    <td>6.1970</td>
    <td>0.528</td>
    <td>0.0280</td>
    <td>2.8170</td>
    <td>24</td>
    <td>27</td>
    <td>18</td>
    <td>18</td>
    <td>6</td>
    <td>21</td>
    <td>19.00</td>
  </tr>
  <tr>
    <td>BRISK_TFEAT</td>
    <td>2.4350</td>
    <td>2.2210</td>
    <td>6.2590</td>
    <td>0.535</td>
    <td>0.0270</td>
    <td>2.8620</td>
    <td>25</td>
    <td>25</td>
    <td>20</td>
    <td>22</td>
    <td>5</td>
    <td>23</td>
    <td>20.00</td>
  </tr>
  <tr>
    <td>BRISK_TFEAT_FLANN</td>
    <td>2.4350</td>
    <td>2.2210</td>
    <td>6.2590</td>
    <td>0.535</td>
    <td>0.0270</td>
    <td>2.8620</td>
    <td>25</td>
    <td>25</td>
    <td>20</td>
    <td>22</td>
    <td>5</td>
    <td>23</td>
    <td>20.00</td>
  </tr>
  <tr>
    <td>ORB2_SOSNET</td>
    <td>2.3960</td>
    <td>2.2170</td>
    <td>6.2880</td>
    <td>0.52</td>
    <td>0.0280</td>
    <td>2.9900</td>
    <td>21</td>
    <td>23</td>
    <td>21</td>
    <td>24</td>
    <td>6</td>
    <td>30</td>
    <td>20.83</td>
  </tr>
  <tr>
    <td>ROOT_SIFT</td>
    <td>2.4360</td>
    <td>2.2340</td>
    <td>6.5610</td>
    <td>0.525</td>
    <td>0.0290</td>
    <td>2.5820</td>
    <td>26</td>
    <td>28</td>
    <td>25</td>
    <td>30</td>
    <td>7</td>
    <td>10</td>
    <td>21.00</td>
  </tr>
  <tr>
    <td>ROOT_SIFT_FLANN</td>
    <td>2.4360</td>
    <td>2.2340</td>
    <td>6.5610</td>
    <td>0.525</td>
    <td>0.0290</td>
    <td>2.5820</td>
    <td>26</td>
    <td>28</td>
    <td>25</td>
    <td>30</td>
    <td>7</td>
    <td>10</td>
    <td>21.00</td>
  </tr>
  <tr>
    <td>CONTEXTDESC</td>
    <td>2.4390</td>
    <td>2.2390</td>
    <td>6.5520</td>
    <td>0.528</td>
    <td>0.0290</td>
    <td>2.5310</td>
    <td>27</td>
    <td>29</td>
    <td>23</td>
    <td>36</td>
    <td>7</td>
    <td>6</td>
    <td>21.33</td>
  </tr>
  <tr>
    <td>CONTEXTDESC_FLANN</td>
    <td>2.4390</td>
    <td>2.2390</td>
    <td>6.5520</td>
    <td>0.528</td>
    <td>0.0290</td>
    <td>2.5310</td>
    <td>27</td>
    <td>29</td>
    <td>24</td>
    <td>36</td>
    <td>7</td>
    <td>6</td>
    <td>21.50</td>
  </tr>
  <tr>
    <td>ORB2</td>
    <td>2.3980</td>
    <td>2.2190</td>
    <td>6.5760</td>
    <td>0.523</td>
    <td>0.0290</td>
    <td>3.2160</td>
    <td>22</td>
    <td>24</td>
    <td>26</td>
    <td>28</td>
    <td>7</td>
    <td>37</td>
    <td>24.00</td>
  </tr>
  <tr>
    <td>AKAZE</td>
    <td>2.4410</td>
    <td>2.2510</td>
    <td>6.6500</td>
    <td>0.565</td>
    <td>0.0500</td>
    <td>2.5700</td>
    <td>28</td>
    <td>31</td>
    <td>28</td>
    <td>42</td>
    <td>8</td>
    <td>8</td>
    <td>24.17</td>
  </tr>
  <tr>
    <td>ALIKED_FLANN</td>
    <td>2.4430</td>
    <td>2.2410</td>
    <td>6.6060</td>
    <td>0.528</td>
    <td>0.0280</td>
    <td>2.8150</td>
    <td>29</td>
    <td>30</td>
    <td>27</td>
    <td>35</td>
    <td>6</td>
    <td>20</td>
    <td>24.50</td>
  </tr>
  <tr>
    <td>XFEAT_XFEAT</td>
    <td>2.4180</td>
    <td>2.2140</td>
    <td>7.0060</td>
    <td>0.532</td>
    <td>0.0280</td>
    <td>3.4530</td>
    <td>23</td>
    <td>22</td>
    <td>36</td>
    <td>20</td>
    <td>6</td>
    <td>40</td>
    <td>24.50</td>
  </tr>
  <tr>
    <td>DISK</td>
    <td>2.4610</td>
    <td>2.2340</td>
    <td>6.7760</td>
    <td>0.521</td>
    <td>0.0290</td>
    <td>2.9500</td>
    <td>31</td>
    <td>28</td>
    <td>31</td>
    <td>25</td>
    <td>7</td>
    <td>27</td>
    <td>24.83</td>
  </tr>
  <tr>
    <td>FAST_FREAK</td>
    <td>2.5370</td>
    <td>2.2250</td>
    <td>6.7990</td>
    <td>0.4208</td>
    <td>0.0320</td>
    <td>4.5670</td>
    <td>38</td>
    <td>26</td>
    <td>32</td>
    <td>2</td>
    <td>9</td>
    <td>43</td>
    <td>25.00</td>
  </tr>
  <tr>
    <td>ORB2_HARDNET_FLANN</td>
    <td>2.4470</td>
    <td>2.2500</td>
    <td>6.6820</td>
    <td>0.522</td>
    <td>0.0290</td>
    <td>2.8860</td>
    <td>30</td>
    <td>32</td>
    <td>30</td>
    <td>27</td>
    <td>7</td>
    <td>25</td>
    <td>25.17</td>
  </tr>
  <tr>
    <td>ORB2_SOSNET_FLANN</td>
    <td>2.4430</td>
    <td>2.2630</td>
    <td>6.6310</td>
    <td>0.527</td>
    <td>0.0280</td>
    <td>3.1340</td>
    <td>29</td>
    <td>33</td>
    <td>29</td>
    <td>33</td>
    <td>6</td>
    <td>35</td>
    <td>27.50</td>
  </tr>
  <tr>
    <td>ORB2_L2NET_FLANN</td>
    <td>2.4890</td>
    <td>2.2980</td>
    <td>6.9130</td>
    <td>0.526</td>
    <td>0.0280</td>
    <td>3.0160</td>
    <td>32</td>
    <td>35</td>
    <td>34</td>
    <td>31</td>
    <td>6</td>
    <td>31</td>
    <td>28.17</td>
  </tr>
  <tr>
    <td>LK_FAST</td>
    <td>2.5290</td>
    <td>2.3330</td>
    <td>6.8440</td>
    <td>0.555</td>
    <td>0.0280</td>
    <td>2.7650</td>
    <td>37</td>
    <td>40</td>
    <td>33</td>
    <td>38</td>
    <td>6</td>
    <td>18</td>
    <td>28.67</td>
  </tr>
  <tr>
    <td>ORB2_HARDNET</td>
    <td>2.4950</td>
    <td>2.3070</td>
    <td>6.9960</td>
    <td>0.526</td>
    <td>0.0280</td>
    <td>3.0460</td>
    <td>33</td>
    <td>37</td>
    <td>35</td>
    <td>32</td>
    <td>6</td>
    <td>32</td>
    <td>29.17</td>
  </tr>
  <tr>
    <td>ORB2_BEBLID</td>
    <td>2.5010</td>
    <td>2.3090</td>
    <td>7.0190</td>
    <td>0.528</td>
    <td>0.0290</td>
    <td>3.0830</td>
    <td>34</td>
    <td>38</td>
    <td>37</td>
    <td>34</td>
    <td>7</td>
    <td>34</td>
    <td>30.67</td>
  </tr>
  <tr>
    <td>ORB2_BEBLID_FLANN</td>
    <td>2.5010</td>
    <td>2.3170</td>
    <td>7.0200</td>
    <td>0.551</td>
    <td>0.0290</td>
    <td>3.1480</td>
    <td>34</td>
    <td>39</td>
    <td>38</td>
    <td>37</td>
    <td>7</td>
    <td>36</td>
    <td>31.83</td>
  </tr>
  <tr>
    <td>XFEAT</td>
    <td>2.5110</td>
    <td>2.2930</td>
    <td>7.4280</td>
    <td>0.556</td>
    <td>0.0290</td>
    <td>3.4710</td>
    <td>35</td>
    <td>34</td>
    <td>39</td>
    <td>39</td>
    <td>7</td>
    <td>41</td>
    <td>32.50</td>
  </tr>
  <tr>
    <td>XFEAT_FLANN</td>
    <td>2.5190</td>
    <td>2.3010</td>
    <td>7.4880</td>
    <td>0.557</td>
    <td>0.0290</td>
    <td>3.4950</td>
    <td>36</td>
    <td>36</td>
    <td>40</td>
    <td>40</td>
    <td>7</td>
    <td>42</td>
    <td>33.50</td>
  </tr>
  <tr>
    <td>ALIKED</td>
    <td>2.5810</td>
    <td>2.3820</td>
    <td>7.5280</td>
    <td>0.574</td>
    <td>0.0280</td>
    <td>3.0580</td>
    <td>40</td>
    <td>42</td>
    <td>41</td>
    <td>43</td>
    <td>6</td>
    <td>33</td>
    <td>34.17</td>
  </tr>
  <tr>
    <td>XFEAT_LIGHTGLUE</td>
    <td>2.5640</td>
    <td>2.3570</td>
    <td>7.9170</td>
    <td>0.561</td>
    <td>0.0500</td>
    <td>3.2420</td>
    <td>39</td>
    <td>41</td>
    <td>42</td>
    <td>41</td>
    <td>8</td>
    <td>38</td>
    <td>34.83</td>
  </tr>
</tbody>
</table>


---
## Step-by-Step Reproduction of Benchmark Results

The following procedure describes how to reproduce the visual odometry benchmark evaluation using the provided configuration files and scripts.

---

### 1. Clone and Set Up PySLAM

Clone the PySLAM repository and initialize all submodules:

```bash
git clone --recursive https://github.com/luigifreda/pyslam.git
cd pyslam
git checkout 9c20866f70b29d1bd0b38ede5592516b7f9502ed
chmod +x install_all_conda.sh
./install_all_conda.sh
```

This script creates a dedicated `pyslam` conda environment and installs all required dependencies.

---

### 2. Download Datasets

The following datasets must be downloaded to replicate the experiments:

- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/
- **EuRoC MAV Dataset**: https://projects.asl.ethz.ch/datasets/doku.php?id=euroc

---

### 3. Integrate Benchmark Scripts

Copy the evaluation scripts and configuration overrides into the appropriate directories within the PySLAM repository:

```bash
# From this repository:
cp main_vo_batch.py /path/to/pyslam/
cp evaluate_all.py /path/to/pyslam/
cp feature_tracker_configs.py /path/to/pyslam/local_features/
```

**Note:** `feature_tracker_configs.py` replaces the default configuration file in `local_features/`.

It is recommended to place `main_vo_batch.py` and `evaluate_all.py` in the PySLAM root directory for simplicity, although custom paths may be used with appropriate adjustments.

---

### 4. Execute Pipelines and Evaluate Results

Select the dataset(s) to evaluate by modifying the `DATASET` variable in `main_vo_batch.py`:

```python
# Possible values: "kitti", "euroc", "both"
DATASET = "both"
```

#### a) Execute all VO pipelines:
```bash
cd /path/to/pyslam
python main_vo_batch.py
```

This step runs all detector-descriptor-matcher configurations on the selected dataset(s) and saves the resulting trajectories.

#### b) Compute evaluation metrics:
```bash
python evaluate_all.py
```

This script calculates the six localization metrics (ATE, MAE, MSE, MRE, RPE, FDE) and outputs the results as CSV files (`kitti_metrics.csv`, `euroc_metrics.csv`) in the current directory.

---

### Modifying or Extending Pipeline Configurations

To add or customize VO configurations:

- Edit `local_features/feature_tracker_configs.py`
- Then re-run:
  ```bash
  python main_vo_batch.py
  python evaluate_all.py
  ```

This will regenerate the trajectories and re-evaluate the metrics accordingly.

---

## Known Issues and Fixes

During evaluation, several descriptor and matcher configurations produced runtime errors due to type mismatches, library incompatibilities, or framework-specific constraints.  
Below, we outline the affected methods and the modifications applied to enable successful execution.

- **BRISK + TFEAT**  
  `cv2.batchDistance` error: type mismatch between `CV_32F` and `CV_8U`  
  **Fix**: Cast TFEAT descriptor output to `np.float32`.

- **KAZE + Kornia**  
  `detectAndCompute()` failed to return descriptors.  
  **Fix**: Preprocess input images with `image_to_tensor()` and add type checks.

- **R2D2**  
  PyTorch `masked_scatter_` error due to float16 vs float32 mismatch.  
  **Fix**: Convert tensors to `.float()`.

- **LFNET**  
  - Issue 1: `masked_scatter_` with float16  
  - Issue 2: Tensor reuse conflict  
  - Issue 3: `tf.AUTO_REUSE` missing (TF1/TF2 compatibility)  
  **Fix**: Use `.float()`, manually reset TF graph and session, avoid mixed TF versions.

- **CONTEXTDESC**  
  `index_put requires matching dtypes` (float16 vs float32)  
  **Fix**: Use `descrs.float()` to unify types.

- **KEYNET / KEYNETAFFNETHARDNET**  
  Error in `extract_patches_from_pyramid()` due to `masked_scatter_` dtype mismatch.  
  **Fix**: Add `.float()` in `detectAndCompute()`.

- **DELF**  
  Error in `ResizeImage()`: `image.ndims` → `image.ndim`  
  **Fix**: Edit `utils.py` to correct property name.

- **LIGHTGLUE + SIFT**  
  `KeyError: 'scales'` during `LightGlue.forward()`  
  **Fix**: Extract `angle` and `scale` from keypoints based on their type; add `.get()` or key checks in LightGlue.

---

### Output Artifacts

After completion, you will find:
- Per-metric CSVs written to the current working directory as `kitti_metrics.csv` and `euroc_metrics.csv`
- These contain ATE, MAE, MSE, MRE, RPE, and FDE values for each sequence and their averages

No plots or additional formatted outputs are generated by default.

---
### Trajectory visualisation helpers

Use the helper scripts in `source_code/` to prepare trajectories for [evo](https://github.com/MichaelGrupp/evo) and quick matplotlib inspection.

- **Convert XYZ keyframes to KITTI poses** – `to_kitti_kf.py` rewrites plain `x y z` tracks into the 12-value KITTI pose format that `evo_traj` expects.
  ```bash
  python source_code/to_kitti_kf.py results/keyframes_xyz.txt results/kf_kitti.txt
  # inspect with evo
  evo_traj kitti dataset/poses/00.txt results/kf_kitti.txt -p --save_plot trajectory_plots/00_kf_evo.png
  ```
  The converter keeps only the translation components; rotations are filled with an identity matrix, which is sufficient for plotting and drift inspection in evo.
- **Plot one or many trajectories with matplotlib** – `trajectory_plot.py` overlays ground truth and VO outputs and supports broadcasting one `--gt` against multiple `--sc` files.
  ```bash
  python source_code/trajectory_plot.py \
    --gt dataset/poses/00.txt \
    --sc results/VO/KITTI/ORB/00.txt results/VO/KITTI/LIGHTGLUESIFT/00.txt \
    --output trajectory_plots/00_orb_vs_lightglue.png --no-show
  ```
  Omit `--no-show` to open an interactive window. The script auto-detects KITTI or XYZ formats and normalises trajectory length before plotting.

---
### Example trajectory plots
<h3>Trajectories from High-Performing Trackers</h3>
<p float="left">
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_00.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_01.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_02.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_04.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_05.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_06.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_07.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_08.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_09.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/good_seq_10.png?raw=true" width="500"/>
</p>

<h3>Trajectories from Low-Performing Trackers</h3>
<p float="left">
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_00.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_01.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_02.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_04.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_05.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_06.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_07.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_08.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_09.png?raw=true" width="500"/>
  <img src="https://github.com/nagyarmand/visual-odometry-benchmarking/blob/main/trajectory_plots/bad_seq_10.png?raw=true" width="500"/>
</p>


---

## References

1. OpenCV Documentation. cv::BFMatcher Class Reference.  
2. Muja, M.; Lowe, D.G. Scalable Nearest Neighbor Algorithms. *IEEE Trans. PAMI*, 2014.  
3. Lucas, B.D.; Kanade, T. An Iterative Image Registration Technique. *Proc. IJCAI*, 1981, pp. 674–679.  
4. Alcantarilla, P.F.; et al. Fast Explicit Diffusion for Accelerated Features. *Proc. BMVC*, 2013.  
5. Zhao, X.; et al. ALIKED: Lightweight Keypoint Extraction. *arXiv*, 2023, arXiv:2304.03608.  
6. Leutenegger, S.; et al. BRISK: Binary Robust Invariant Scalable Keypoints. *Proc. ICCV*, 2011, pp. 2548–2555.  
7. Balntas, V.; Riba, E.; Ponsa, D.; Mikolajczyk, K. Learning Local Feature Descriptors with Triplets and Shallow Convolutional Neural Networks. In *Proc. BMVC*, 2016.  
8. Luo, Z.; et al. ContextDesc: Descriptor Augmentation. *arXiv*, 2019, arXiv:1904.04084.  
9. Dusmanu, M.; et al. D2-Net: A Trainable CNN for Joint Description and Detection of Local Features. *Sci. Direct – Robotics and Autonomous Systems*, 2022.  
10. Tyszkiewicz, M.; et al. DISK: Learning Local Features with Policy Gradient. *arXiv*, 2020, arXiv:2006.13566.  
11. Rosten, E.; Porter, R.; Drummond, T. Faster and Better: A Machine Learning Approach to Corner Detection. arXiv preprint arXiv:2012.00859, 2020.  
12. Alahi, A.; et al. FREAK: Fast Retina Keypoint. *Proc. CVPR*, 2012.  
13. Rublee, E.; et al. ORB: An Efficient Alternative to SIFT or SURF. *Proc. ICCV*, 2011, pp. 2564–2571.  
14. Alcantarilla, P.F.; et al. KAZE Features. *Proc. ECCV*, 2012.  
15. Barroso-Laguna, A.; et al. Key.Net: Keypoint Detection via CNN Filters. *arXiv*, 2019, arXiv:1904.00889.  
16. Wang, Y. Silver Medal Solution for Image Matching Challenge 2024. *arXiv*, 2024, arXiv:2411.01851.  
17. Ono, Y.; Trulls, E.; Fua, P.; Yi, K.M. LF-Net: Learning Local Features from Images. *arXiv*, 2018, arXiv:1805.09662.  
18. DeTone, D.; et al. SuperPoint: Self-Supervised Interest Point Detection and Description. *arXiv*, 2018, arXiv:1712.07629.  
19. Lindenberger, D.; et al. LightGlue: Feature Matching at Light Speed. *arXiv*, 2023, arXiv:2204.09103.  
20. Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. *Int. J. Comput. Vis.*, 2004, 60(2), 91–110.  
21. Shi, J.; Tomasi, C. Good Features to Track. *Proc. CVPR*, 1994.  
22. Mur-Artal, R.; Tardós, J.D. ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras. *IEEE Trans. Robot.*, 2017, 33(5), 1255–1262.  
23. Chao, P.; et al. HarDNet: A Low Memory Traffic Network. *arXiv*, 2019, arXiv:1909.00948.  
24. Tian, Y.; Fan, B.; Wu, F. L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space. *Proc. CVPR*, 2017, pp. 6128–6136.  
25. Tian, Y.; et al. SOSNet: Second Order Similarity Regularization. *arXiv*, 2019, arXiv:1904.05019.  
26. Revaud, J.; Glaunès, J. R2D2: Repeatable and Reliable Detector and Descriptor. *arXiv*, 2019, arXiv:1906.06195.  
27. Arandjelović, R.; Zisserman, A. Three Things to Improve Object Retrieval. *Proc. CVPR*, 2012, pp. 2911–2918.  
28. Potje, G.; Cadar, F.; Araujo, A.; Martins, R.; Nascimento, E.R. XFeat: Accelerated Features for Lightweight Image Matching. *arXiv*, 2024, arXiv:2404.19174.
