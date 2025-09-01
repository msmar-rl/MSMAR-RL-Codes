# MSMAR-RL: Multi-Step Masked-Attention Recovery Reinforcement Learning for Safe Maneuver Decision in High-Speed Pursuit-Evasion Game

### Paper Link

[https://msmar-rl.github.io/](https://msmar-rl.github.io/)

### Environment Setup

- **Create Conda Environment**

```sh
conda create -n msmar-rl python==3.8
conda activate msmar-rl
```

- **Install Isaac Gym**

Download *Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release* from [Isaac Gym - Download Archive | NVIDIA Developer](https://developer.nvidia.com/isaac-gym/download), and install it in the created Conda environment.

```sh
cd isaacgym/python 
pip install -e .
```

After installation, you can verify success by running the following demo under `isaacgym/python/examples`:

```sh
python joint_monkey.py
```

If the simulation window opens, the installation is successful.

- **Install Isaac Gym Envs**

After Isaac Gym is installed, install its environment package:

```sh
cd IsaacGymEnvs
pip install -e .
```

- **Install skrl-1.0.0**

The proposed algorithm is implemented on the **skrl-1.0.0** framework (open-sourced). Install it as follows:

```sh
cd skrl-1.0.0
pip install -e .["torch"]
```

---

### Reproducing Results

- **Modify Absolute Path Dependencies**

  - **UAV Model Loading**  
    In the file `submit_version/IsaacGymEnvs/isaacgymenvs/tasks/UAV_multi_obstacle_recovery_test.py`, modify the absolute paths at lines **386, 396, 402, 408, 414, 420, 426** to the absolute path where your cloned repository is stored.

  - **Flight Trajectory Saving Path**  
    In the same file, modify the absolute paths at lines **61, 69, 70** to your repository path.

- **Run Test Code**

  ```sh
  cd IsaacGymEnvs/isaacgymenvs
  python test_recovery_risk_judge.py
  ```

  - During execution, the console will continuously output the UAV’s distance to each obstacle, allowing you to determine whether the UAV enters a danger zone.  
  - After execution, the UAV flight trajectory will be saved at:  
    ```
    /submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi/test_recovery_1.txt.acmi
    ```  
    This file can be replayed using **TacView** to visualize the trajectory.  
  - Additionally, the file `record_safe.txt.acmi` will also be saved in the same directory. This file (viewable in plain text) logs the UAV’s real-time distances to danger zones.
