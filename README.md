# IsaacGraspEnv

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository contains environments and agents for learning strategies for grasping objects with a dexterous hand. The repository have 9 environments with the same world, reward, and differing vectors of action and observation.
There are three types of observation:
- Observation with oracle information about the object. Environment name prefix is `Cube`.
- Observation with a sampled point cloud on the object. Environment name prefix is `Full-Obj-PC`.
- Observation with a point cloud built by the camera. Environment name prefix is `PC`.

And three types of action space:
- Vector of target positions in the robot's configuration space.
- Vector of target positions for the end effector in Cartesian space. Environment name prefix is `IK-Abs`.
- Vector of end effector displacement in Cartesian space. Environment name prefix is `IK-Rel`.

Below is the environment with the DP-Flex arm and the kuka iiwa 14 manipulator. The agent's task is to grasp the object and move the target position:

- Action: joint position of manipulator
    - `Isaac-Lift-Cube-Iiwa-v0` - Observation: object's oracle information,
    - `Isaac-Full-Obj-PC-Lift-Iiwa-v0` - Observation: a sampled point cloud on the object, 
    - `Isaac-PC-Lift-Iiwa-v0` - Observation: a point cloud built by the camera
- Action: absolute pose end-effector of manipulator in Cartesian space
    - `Isaac-Lift-Cube-Iiwa-IK-Abs-v0` - Observation: object's oracle information,
    - `Isaac-Full-Obj-PC-Lift-Iiwa-IK-Abs-v0` - Observation: a sampled point cloud on the object, 
    - `Isaac-PC-Lift-Iiwa-IK-Abs-v0` - Observation: a sampled point cloud on the object.
- Action: relative pose end-effector of manipulator in Cartesian space
    - `Isaac-Lift-Cube-Iiwa-IK-Rel-v0` - Observation: object's oracle information,
    - `Isaac-Full-Obj-PC-Lift-Iiwa-IK-Rel-v0` - Observation: a sampled point cloud on the object, 
    - `Isaac-PC-Lift-Iiwa-IK-Rel-v0` - Observation: a sampled point cloud on the object.

Full description of the environment below.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/BE2R-Lab-RND-AI-Grasping/IsaacGraspingEnv.git

# Option 2: SSH
git clone git@github.com:BE2R-Lab-RND-AI-Grasping/IsaacGraspingEnv.git
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/IsaacGraspEnv
```

- Download dataset of objects and extract zip: [link](https://disk.yandex.ru/d/NPZwTWIzPU2gvg)

- Run scripts for converting obj files in usd:

```bash
python ./scripts/dataset_loader/dataset_usd_converter.py --dataset_path <PATH_TO_DATASET> --name_point_cloud point_cloud_colorless.ply --obj_file_name object_convex_decomposition_meter_unit.obj --path_usd <PATH_TO_USD>
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/autonomous_agent.py --dataset_path <PATH_TO_USD>/power_drills --model_filter "1, 5"
```
This environment presents an autonomous agent without training using a finite state machine.

## Scripts

Runs policy learning in the `Isaac-Lift-Cube-Iiwa-IK-Rel-v0` environment using the SKRL library:

```bash
python scripts/skrl_train.py
```
Starts testing of the trained policy

```bash
python scripts/skrl_play.py
```
## Environment description

### Observation Space

Observation Space

The observation space includes:

- $H_\text{ee}^b$ - Robot's end-effector (hand wrist frame) pose relative to the robot's base frame
- $\mathbf{p}^b_i, i \in \text{Fingers}$ - Positions of fingertips in the base frame
- $\mathbf{q}, \dot{\mathbf{q}}$ - Joint positions and velocities of the robot
- $\mathbf{p}^b_\text{t}$ - Target position of the object in the base frame
- $\mathbf{a}_{t-1}$ - Last action taken by the robot

Information about objects and scene supports three types of observation:

- Oracle Information: $H_\text{o}^b$ - Object pose in the base frame
- Sampled Point Cloud: $\mathbf{p}^b_\text{o}$ - Set of points sampled from the object mesh in the base frame
- Camera-based Point Cloud: $\mathbf{p}^b_\text{o}$ - Set of points from the object captured by simulated depth cameras in the base frame

## Project Structure
```
├── scripts/                # Training, evaluation and utility scripts
│   ├── autonomous_agent.py # Scripts runs autonomous agent without training using a finite state machine
│   ├── skrl_play.py        # Scripts runs last trained policy
│   ├── skrl_train.py       # Scripts start trainig policy process
├── logs/                   # Training logs and experiment results
├── source/IsaacGraspEnv/IsaacGraspEnv
│   ├── assets/             # Assets of objects for grasping
│   ├── robots/             # USD description files of robots
│   ├── sim/                # Extended config classes and schemas from IsaacLab
│   ├── tasks/              # RL environments and task definitions
│   ├── torch_models/       # Neural network architectures for policies and feature extraction
│   └── dataset_managers/   # Tools for loading and managing external object datasets
├── README.md
└── requirements.txt
```
## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/ext_template"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
