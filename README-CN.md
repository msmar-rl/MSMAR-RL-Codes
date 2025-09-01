
<h1 align="center"> MSMAR-RL: Multi-Step Masked-Attention Recovery Reinforcement Learning for Safe Maneuver Decision in High-Speed Pursuit-Evasion Game</h1>

### 论文链接

https://msmar-rl.github.io/

### 环境配置

- 创建conda环境

```sh
conda create -n msmar-rl python==3.8
conda activate msmar-rl
```

- Isaac Gym 安装

在[Isaac Gym - Download Archive | NVIDIA Developer](https://developer.nvidia.com/isaac-gym/download)中，下载Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release，并在满足要求的conda环境中，安装Isaac Gym

```sh
cd isaacgym/python 
pip install -e .
```

安装完成后，可以通过在 isaacgym/python/examples路径下，运行下述代码，若能够开启仿真界面，说明安装成功。

```sh
python joint_monkey.py
```

- Isaac Gym Envs安装

完成Isaac Gym安装后，安装与之配套的训练环境。

```sh
cd msmar/IsaacGymEnvs
pip install -e .
```

- 安装skrl-1.0.0

本研究实现的算法，基于skrl-1.0.0算法平台，代码已开源，安装指令如下。

```sh
cd msmar/skrl-1.0.0
pip install -e .["torch"]
```

### 结果复现

- 修改绝对路径依赖
  - 无人机模型载入
    - 在 "msmar/IsaacGymEnvs/isaacgymenvs/tasks/UAV_multi_obstacle_recovery_test.py"文件中的line 386,396,402,408,414,420,426行，修改绝对路径为msmar/IsaacGymEnvs/isaacgymenvs/tasks文件夹存放的绝对路径；

  - 飞行轨迹保存路径
    - 在 "msmar/IsaacGymEnvs/isaacgymenvs/tasks/UAV_multi_obstacle_recovery_test.py"文件中的line61,69,70行，修改绝对路径为msmar/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi/文件夹存放的绝对路径。


- 运行测试代码

  - 运行下列代码

  ```sh
  cd IsaacGymEnvs/isaacgymenvs
  python test_recovery_risk_judge.py
  ```

  - 运行该文件后，控制台会实时输出无人机与各个障碍物的距离，便于确定无人机是否进入危险区域。
  - 结束运行后，无人机飞行轨迹会保存在 “/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi/test_recovery_1.txt.acmi” 文件中，该文件可通过 TacView软件进行轨迹回放，观察无人机轨迹。此外，该目录下同时保存了 record_safe.txt.acmi 文件，该文件可通过txt查看，记录了无人机与危险区域的实时距离。

