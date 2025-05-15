# IsaacGraspEnv

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository contains environments and agents for learning strategies for grasping objects with a dexterous hand. 
Accessible Environments:
- `Isaac-Lift-Cube-Iiwa-IK-Rel-v0` - environment with DP-Flex grasp and manipulator iiwa14. The agent's task is to grab the screwdriver and move the target position.

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
