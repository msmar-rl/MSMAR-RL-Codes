# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import copy
import math
import numpy as np
import os
import csv
import torch
import xml.etree.ElementTree as ET
import random
from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
from .Eva_tools_multi import *

from pyproj import Geod, CRS, Transformer


from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.torch import load_isaacgym_env_preview4
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed



#################################

def set_excel(episode):
    title_tac = ['Time', 'Latitude', 'Longitude', 'Altitude', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    
    acmi_tac = open(f'/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi/test_recovery_1.txt.acmi', 'w', encoding='UTF8')
    # acmi_tac = open(f'/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi_cylinder/test_recovery_1.txt.acmi', 'w', encoding='UTF8')

    acmi_tac.write("FileType=text/acmi/tacview\n")
    acmi_tac.write("FileVersion=2.1\n")
    acmi_tac.write("0,ReferenceTime=2024-07-01T00:00:00Z\n")
    
    # multi obstacles
    acmi_writer = open(f'/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi/test_recovery_1.txt.acmi', 'a', encoding='UTF8')
    safe_writer = open(f'/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi/record_safe.txt.acmi', 'w', encoding='UTF8')

    # multi cylinders
    # acmi_writer = open(f'/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi_cylinder/test_recovery_1.txt.acmi', 'a', encoding='UTF8')
    # safe_writer = open(f'/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/data/ddpg_multi_cylinder/record_safe.txt.acmi', 'w', encoding='UTF8')

    return acmi_writer,safe_writer

def logmsg(id,name,color,lon,lat,alt,roll,pitch,yaw):
    log_msg = f"{id},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
    log_msg+= f"Name={name},"
    log_msg+= f"Color={color}\n"
    return log_msg

def MercatorTOWGS84(x, y):
    crs_WGS84 = CRS.from_epsg(4326)    
    crs_WebMercator = CRS.from_epsg(3857)

    transformer = Transformer.from_crs(crs_WebMercator, crs_WGS84)
    lat, lon = transformer.transform(x, y)
    return lat, lon



# ji suan liang xiang liang jia jiao
def get_angle_between_vectors(v1, v2):
    v1_normalized = v1 / torch.norm(v1, dim=1, keepdim=True)
    v2_normalized = v2 / torch.norm(v2, dim=1, keepdim=True)

    cosine_similarities = torch.sum(v1_normalized * v2_normalized, dim=1)

    cosine_similarities = torch.clamp(cosine_similarities, min=-1.0, max=1.0)

    cosine_similarities[torch.isnan(cosine_similarities)] = 0.0
    cosine_similarities[torch.isinf(cosine_similarities)] = 0.0

    angle = torch.acos(cosine_similarities)

    angle = angle * (180.0 / math.pi)

    return angle


######## si yuan shu zhuan hua xiang liang ########
def rotate_voctor_with_quaternion(vector, quaternion):
    norm = torch.linalg.norm(quaternion)
    quaternion = quaternion / norm

    vec_quat = torch.zeros(4).to(vector.device)
    vec_quat[:3] = vector

    quat_conj = quaternion.clone()
    quat_conj[:3] *= -1

    temp = quaternion_mult(quaternion, vec_quat)
    rotated_vec_quat = quaternion_mult(temp, quat_conj)

    return rotated_vec_quat[:3]


def quaternion_mult(q1, q2):
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.tensor([x, y, z, w]).to(q1.device)


# 计算plane and itself xyz angle
def calculate_angle(quaternion, device):
    angle = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float)

    x = rotate_voctor_with_quaternion(x_axis, quaternion)
    y = rotate_voctor_with_quaternion(y_axis, quaternion)
    z = rotate_voctor_with_quaternion(z_axis, quaternion)

    angle[0] = get_angle_between_vectors(x, x_axis)
    angle[1] = get_angle_between_vectors(y, y_axis)
    angle[2] = get_angle_between_vectors(z, z_axis)

    return angle


# set plane's x zhou is level (xuan zhuan shi ji plane)
def rotate_to_horizontal(plane_quaternion, device):
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float)
    x = rotate_voctor_with_quaternion(x_axis, plane_quaternion)

    angle = torch.atan2(torch.tensor([x[2]], device=device, dtype=torch.float),
                        torch.tensor([x[1]], device=device, dtype=torch.float))

    rotation_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float)
    rotation_quaternion = torch.tensor([torch.cos(torch.tensor([angle / 2], device=device, dtype=torch.float)),
                                        rotation_axis[0] * torch.sin(
                                            torch.tensor([angle / 2], device=device, dtype=torch.float)),
                                        rotation_axis[1] * torch.sin(
                                            torch.tensor([angle / 2], device=device, dtype=torch.float)),
                                        rotation_axis[2] * torch.sin(
                                            torch.tensor([angle / 2], device=device, dtype=torch.float))
                                        ], device=device, dtype=torch.float)

    w1, x1, y1, z1 = rotation_quaternion
    x2, y2, z2, w2 = plane_quaternion
    new_plane_quaternion = torch.tensor([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,  # y
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,  # z
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
    ], device=device)

    new_plane_quaternion /= torch.norm(new_plane_quaternion)

    return new_plane_quaternion, angle.item()


class UAV_multi_obstacle_recovery_test(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.flog = 0
        self.time = 0
        self.acmi_writer,self.safe_writer = set_excel(self.flog)
        
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        states_per_actor = 13
        # states_target_actor = 13 #no 8 vec, only 8 pos none
        rigids_per_actor = 1
        # force_per_actor = 4
        control = 3

        object = 6
        actors = 1
        roots = object + actors  # 2+1
        rigids = object + rigids_per_actor * actors  # 2+9*1

        # Observations:

        obs_per_actor = states_per_actor * 0 + 13

        observations = 13+6  # 13+position
        # observations += two_plane_interaction

        # Actions:

        control_per_actor = 3
        actions = control_per_actor * actors

        # self.cfg["env"]["num_observations_discrete"] = 13
        self.cfg["env"]["numObservations"] = observations
        self.cfg["env"]["numActions"] = actions
        self.cfg["env"]["numActions_discrete"] = 15
        self.num_obs_per_actor = obs_per_actor
        self.num_actors = actors
        self.num_object = object

        # create_sim()
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, roots,
                                                                      states_per_actor)  # env*3*13

        self.root_states = vec_root_tensor

        self.quadcopter_positions = []
        self.quadcopter_quats = []
        self.quadcopter_linvels = []
        self.quadcopter_angvels = []
        for i in range(actors):
            self.quadcopter_positions.append(vec_root_tensor[..., i, 0:3])
            self.quadcopter_quats.append(vec_root_tensor[..., i, 3:7])
            self.quadcopter_linvels.append(vec_root_tensor[..., i, 7:10])
            self.quadcopter_angvels.append(vec_root_tensor[..., i, 10:13])

        self.rigid_positions = []
        self.rigid_quats = []
        self.rigid_linvels = []
        self.rigid_angvels = []
        for i in range(object):
            ii = i + actors
            self.rigid_positions.append(vec_root_tensor[..., ii, 0:3])
            self.rigid_quats.append(vec_root_tensor[..., ii, 3:7])
            self.rigid_linvels.append(vec_root_tensor[..., ii, 7:10])
            self.rigid_angvels.append(vec_root_tensor[..., ii, 10:13])

        self.dof_positions = []
        self.dof_velocities = []

        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = vec_root_tensor.clone()
        # opposite
        self.initial_root_states[:, 0, 0:3] = torch.zeros(1, dtype=torch.float32).repeat(self.num_envs, 3)
        # obstacle
        self.initial_root_states[:, 1, 0:3] = torch.zeros(1, dtype=torch.float32).repeat(self.num_envs, 3)
        # self.initial_root_states[:, 3, 0:3] = torch.zeros(1, dtype=torch.float32).repeat(self.num_envs, 3)
        # self.initial_root_states[:, 4, 0:3] = torch.zeros(1, dtype=torch.float32).repeat(self.num_envs, 3)
        # self.initial_root_states[:, 5, 0:3] = torch.zeros(1, dtype=torch.float32).repeat(self.num_envs, 3)
        # self.initial_root_states[:, 6, 0:3] = torch.zeros(1, dtype=torch.float32).repeat(self.num_envs, 3)
        
        

        self.v_lower_limits = 0.5 * torch.ones(1, device=self.device, dtype=torch.float32)
        self.v_upper_limits = 4.0 * torch.ones(1, device=self.device, dtype=torch.float32)

        self.g_lower_limits = - math.pi / 2.0 * torch.ones(1, device=self.device, dtype=torch.float32)
        self.g_upper_limits = math.pi / 2.0 * torch.ones(1, device=self.device, dtype=torch.float32)

        self.p_lower_limits = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.p_upper_limits = math.pi * 2 * torch.ones(1, device=self.device, dtype=torch.float32)

        self.all_actor_indices = torch.arange(roots * self.num_envs, dtype=torch.int32, device=self.device).view(
            self.num_envs, roots)
        self.all_ship_indices = (
                roots * torch.arange(self.num_envs, dtype=torch.int32, device=self.device) + roots - 1).view(
            self.num_envs)

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, rigids, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

        ###################### dong li xue mo xing can shu #############################
        # miu
        self.red_miu = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)

        # Initialization reard
        self.miu_reward = 0.0
        self.blue_miu_reward = 0.0
        self.energy_reward = 0.0

        # blue control
        self.blue_nx = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        self.blue_nz = torch.tensor([1.0/math.cos(math.pi/4.0)], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        self.blue_miu = torch.tensor([math.pi/4.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)

        self.dmiu = 0.0
        self.g = 9.81

        # v
        self.blue_v = torch.tensor([2.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        self.red_v = torch.tensor([2.8], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)

        # gamma
        self.blue_gamma = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        self.red_gammas = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        # self.red_gammas = (random.uniform(-math.pi/6.0, math.pi/6.0) *
        #                    torch.ones(1, device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1))

        # psi
        self.blue_psi = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        self.red_psis = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1)
        

        # self.blue_psi = (random.uniform(-math.pi, math.pi) *
        #                  torch.tensor([1.0], device=self.device,dtype=torch.float32).repeat(self.num_envs, actors, 1))
        # self.red_psis = (random.uniform(-math.pi, math.pi) *
        #                  torch.ones(1, device=self.device, dtype=torch.float32).repeat(self.num_envs, actors, 1))

        self.is_opposite = False
        ################################################################################
        pos_linvels = self.root_states[:, :, 0:3]
        quaternion_linvels = self.root_states[:, :, 3:7]
        pos = pos_linvels.clone()
        quaternion = quaternion_linvels.clone()

        self.red_uav = Uavs_red(pos[:, 0:1, 0:3], self.red_v, self.red_gammas, self.red_psis,
                                self.dt, self.num_envs, self.device, quaternion[:, 0:1, 3:7])

        self.blue_uav = Uavs_blue(pos[:, 1:2, 0:3], self.blue_v, self.blue_gamma, self.blue_psi,
                                  self.dt, self.num_envs, self.device, quaternion[:, 1:2, 3:7])


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = 0
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        # self._create_quadcopter_asset()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # asset_root = "."
        # asset_file = "quadcopter.xml"

        asset_root = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        asset_file = "jet.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        # asset_options.angular_damping = 0.0
        # asset_options.max_angular_velocity = 4 * math.pi
        # asset_options.slices_per_cylinder = 10
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        ship_root1 = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        ship_file1 = "jet.urdf"
        ship_options1 = gymapi.AssetOptions()
        ship_options1.fix_base_link = False
        ship_asset1 = self.gym.load_asset(self.sim, ship_root1, ship_file1, ship_options1)

        ship_root2 = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        ship_file2 = "jet.urdf"
        ship_options2 = gymapi.AssetOptions()
        ship_options2.fix_base_link = False
        ship_asset2 = self.gym.load_asset(self.sim, ship_root2, ship_file2, ship_options2)

        ship_root3 = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        ship_file3 = "jet.urdf"
        ship_options3 = gymapi.AssetOptions()
        ship_options3.fix_base_link = False
        ship_asset3 = self.gym.load_asset(self.sim, ship_root3, ship_file3, ship_options3)

        ship_root4 = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        ship_file4 = "jet.urdf"
        ship_options4 = gymapi.AssetOptions()
        ship_options4.fix_base_link = False
        ship_asset4 = self.gym.load_asset(self.sim, ship_root4, ship_file4, ship_options4)

        ship_root5 = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        ship_file5 = "jet.urdf"
        ship_options5 = gymapi.AssetOptions()
        ship_options5.fix_base_link = False
        ship_asset5 = self.gym.load_asset(self.sim, ship_root5, ship_file5, ship_options5)

        ship_root6 = "/data/zwz/submit_uav/submit_version/IsaacGymEnvs/isaacgymenvs/tasks"
        ship_file6 = "jet.urdf"
        ship_options6 = gymapi.AssetOptions()
        ship_options6.fix_base_link = False
        ship_asset6 = self.gym.load_asset(self.sim, ship_root6, ship_file6, ship_options6)

        ocean_options = gymapi.AssetOptions()
        ocean_options.fix_base_link = True
        ocean_asset = self.gym.create_box(self.sim, 40, 40, 0.01, ocean_options)

        default_pose = gymapi.Transform()

        self.envs = []
        self.obj_handles = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = []
            for j in range(self.num_actors):
                # quadcopter_x = random.uniform(- 5.0, 5.0)
                # quadcopter_y = random.uniform(-15.0, -5.0)
                # quadcopter_z = random.uniform(45.0, 55.0)
                # default_pose.p = gymapi.Vec3(quadcopter_x, quadcopter_y, quadcopter_z)
                # actor_handle.append(self.gym.create_actor(env, asset, default_pose, "quadcopter", i, 0, 0))
                # self.gym.set_actor_scale(env, actor_handle[j], 0.0005)

                quadcopter_x = 3.0
                quadcopter_y = -10.0
                quadcopter_z = 50.0
                default_pose.p = gymapi.Vec3(quadcopter_x, quadcopter_y, quadcopter_z)
                actor_handle.append(self.gym.create_actor(env, asset, default_pose, "quadcopter", i, 0, 0))
                self.gym.set_actor_scale(env, actor_handle[j], 0.0005)

            ship_pose1 = gymapi.Transform()
            # target_z = random.uniform(40.0, 60.0)
            target_z = 50.0
            ship_pose1.p = gymapi.Vec3(- 0.0, - 0.0, target_z)
            ship_handle1 = self.gym.create_actor(env, ship_asset1, ship_pose1, "ship1", i, 0, 0)
            self.obj_handles.append(ship_handle1)
            self.gym.set_actor_scale(env, ship_handle1, 0.0005)

            ship_pose2 = gymapi.Transform()
            # ship_pose2.p = gymapi.Vec3(-20.0, 10.0, 60)
            ship_2_x = random.uniform(-25.0, -15.0)
            ship_2_y = random.uniform(5.0, 15.0)
            ship_2_z = random.uniform(55.0, 65.0)
            ship_pose2.p = gymapi.Vec3(ship_2_x,ship_2_y,ship_2_z)
            ship_handle2 = self.gym.create_actor(env, ship_asset2, ship_pose2, "ship2", i, 0, 0)
            self.obj_handles.append(ship_handle2)
            self.gym.set_actor_scale(env, ship_handle2, 0.0005)


            ship_pose3 = gymapi.Transform()
            # ship_pose3.p = gymapi.Vec3(20.0, -10.0, 55)
            ship_3_x = random.uniform(15.0, 25.0)
            ship_3_y = random.uniform(-5.0, -15.0)
            ship_3_z = random.uniform(50.0, 60.0)
            ship_pose3.p = gymapi.Vec3(ship_3_x,ship_3_y,ship_3_z)
            ship_handle3 = self.gym.create_actor(env, ship_asset3, ship_pose3, "ship3", i, 0, 0)
            self.obj_handles.append(ship_handle3)
            self.gym.set_actor_scale(env, ship_handle3, 0.0005)


            ship_pose4 = gymapi.Transform()
            # ship_pose4.p = gymapi.Vec3(0.0, -1.0, 50)
            ship_4_x = random.uniform(-5.0, 5.0)
            ship_4_y = random.uniform(-5.0, 1.0)
            ship_4_z = random.uniform(45.0, 55.0)
            ship_pose4.p = gymapi.Vec3(ship_4_x,ship_4_y,ship_4_z)
            ship_handle4 = self.gym.create_actor(env, ship_asset4, ship_pose4, "ship4", i, 0, 0)
            self.obj_handles.append(ship_handle4)
            self.gym.set_actor_scale(env, ship_handle4, 0.0005)

            ship_pose5 = gymapi.Transform()
            # ship_pose5.p = gymapi.Vec3(10.0, 10.0, 45)
            ship_5_x = random.uniform(5.0, 15.0)
            ship_5_y = random.uniform(5.0, 15.0)
            ship_5_z = random.uniform(40.0, 50.0)
            ship_pose5.p = gymapi.Vec3(ship_5_x,ship_5_y,ship_5_z)
            ship_handle5 = self.gym.create_actor(env, ship_asset5, ship_pose5, "ship5", i, 0, 0)
            self.obj_handles.append(ship_handle5)
            self.gym.set_actor_scale(env, ship_handle5, 0.0005)

            ship_pose6 = gymapi.Transform()
            # ship_pose6.p = gymapi.Vec3(-10.0, -10.0, 40)
            ship_6_x = random.uniform(-15.0, -5.0)
            ship_6_y = random.uniform(-15.0, -5.0)
            ship_6_z = random.uniform(35.0, 45.0)
            ship_pose6.p = gymapi.Vec3(ship_6_x,ship_6_y,ship_6_z)
            
            ship_handle6 = self.gym.create_actor(env, ship_asset6, ship_pose6, "ship6", i, 0, 0)
            self.obj_handles.append(ship_handle6)
            self.gym.set_actor_scale(env, ship_handle6, 0.0005)

            

            chassis_color = gymapi.Vec3(0.8, 0.6, 0.2)
            for j in range(self.num_actors):
                self.gym.set_rigid_body_color(env, actor_handle[j], 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color)

            self.envs.append(env)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 4 * self.num_actors, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    def pre_physics_step(self, _actions):
        # update pos
        pos_linvels = self.root_states[:, :, 0:3]
        pos = pos_linvels.clone()
        self.red_uav.update_pos(pos[:, 0:1, 0:3])
        self.blue_uav.update_pos(pos[:, 1:2, 0:3])

        # action
        # actions = _actions.to(self.device)
        actions = _actions
        
        bound = torch.tensor([1.5, 4.0, math.pi / 4.0], device=self.device, dtype=torch.float32)
        shift = torch.tensor([0.5, 4.0, 0.0], device=self.device, dtype=torch.float32)

        actions = actions * bound + shift

        # move blue
        # if timestep < per_timestep_change:
        #     blue_actions = self.blue_uav.select_action(self.red_uav, self.blue_uav)
            # blue_actions = self.blue_uav.select_better_action(self.red_uav, self.blue_uav, actions)
        
        blue_actions = self.blue_uav.select_action(self.red_uav, self.blue_uav)        
        blue_actions = self.blue_uav.get_real_move(self.blue_uav, blue_actions)
        self.blue_uav.move(blue_actions)
        blue_v = self.blue_uav.vv

        # move red
        self.red_uav.move(actions)
        red_v = self.red_uav.vv

        # writer data
        red_gammas = self.red_uav.gammas[0:1,:].cpu().numpy() * 180 / np.pi
        red_psis = self.red_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        red_miu = actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        red_pos_x,red_pos_y = MercatorTOWGS84(pos[0:1, 0:1, 0:1].cpu().numpy()*100, pos[0:1, 0:1, 1:2].cpu().numpy()*100)
        red_pos_z = pos[0:1, 0:1, 2:3].cpu().numpy() * 100


        blue_gammas = self.blue_uav.gammas[0:1,:].cpu().cpu().numpy() * 180 / np.pi
        blue_psis = self.blue_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        blue_miu = blue_actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        blue_pos_x,blue_pos_y = MercatorTOWGS84(pos[0:1, 1:2, 0:1].cpu().numpy()*100, pos[0:1, 1:2, 1:2].cpu().numpy()*100)
        blue_pos_z = pos[0:1, 1:2, 2:3].cpu().numpy() * 100

        obstacle_1_gammas = self.blue_uav.gammas[0:1,:].cpu().cpu().numpy() * 180 / np.pi
        obstacle_1_psis   = self.blue_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        obstacle_1_miu    = blue_actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        obstacle_1_x,obstacle_1_y = MercatorTOWGS84(pos[0:1,2:3,0:1].cpu().numpy()*100,pos[0:1,2:3,1:2].cpu().numpy()*100)
        obstacle_1_z      = pos[0:1,2:3,2:3].cpu().numpy()*100

        obstacle_2_gammas = self.blue_uav.gammas[0:1,:].cpu().cpu().numpy() * 180 / np.pi
        obstacle_2_psis   = self.blue_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        obstacle_2_miu    = blue_actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        obstacle_2_x,obstacle_2_y = MercatorTOWGS84(pos[0:1,3:4,0:1].cpu().numpy()*100,pos[0:1,3:4,1:2].cpu().numpy()*100)
        obstacle_2_z      = pos[0:1,3:4,2:3].cpu().numpy()*100

        obstacle_3_gammas = self.blue_uav.gammas[0:1,:].cpu().cpu().numpy() * 180 / np.pi
        obstacle_3_psis   = self.blue_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        obstacle_3_miu    = blue_actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        obstacle_3_x,obstacle_3_y = MercatorTOWGS84(pos[0:1,4:5,0:1].cpu().numpy()*100,pos[0:1,4:5,1:2].cpu().numpy()*100)
        obstacle_3_z      = pos[0:1,4:5,2:3].cpu().numpy()*100

        obstacle_4_gammas = self.blue_uav.gammas[0:1,:].cpu().cpu().numpy() * 180 / np.pi
        obstacle_4_psis   = self.blue_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        obstacle_4_miu    = blue_actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        obstacle_4_x,obstacle_4_y = MercatorTOWGS84(pos[0:1,5:6,0:1].cpu().numpy()*100,pos[0:1,5:6,1:2].cpu().numpy()*100)
        obstacle_4_z      = pos[0:1,5:6,2:3].cpu().numpy()*100

        obstacle_5_gammas = self.blue_uav.gammas[0:1,:].cpu().cpu().numpy() * 180 / np.pi
        obstacle_5_psis   = self.blue_uav.psis[0:1,:].cpu().numpy() * 180 / np.pi
        obstacle_5_miu    = blue_actions[0:1,2:3].cpu().numpy() * 180 / np.pi
        obstacle_5_x,obstacle_5_y = MercatorTOWGS84(pos[0:1,6:7,0:1].cpu().numpy()*100,pos[0:1,6:7,1:2].cpu().numpy()*100)
        obstacle_5_z      = pos[0:1,6:7,2:3].cpu().numpy()*100
        # self.red_writer.writerow(
        #     [self.time, red_pos_x.item() + 10.0, red_pos_y.item() + 10.0, red_pos_z.item(), red_miu.item(), red_gammas.item(), red_psis.item()])
        # self.blue_writer.writerow(
        #     [self.time, blue_pos_x.item() + 10.0, blue_pos_y.item() + 10.0, blue_pos_z.item(), blue_miu.item(), blue_gammas.item(), blue_psis.item()])

        self.acmi_writer.write(f"#{self.time:.2f}\n")
        self.acmi_writer.write(logmsg("A01","F16","Red",red_pos_y.item() + 10.0, red_pos_x.item() + 10.0, red_pos_z.item(), red_miu.item(), red_gammas.item(), red_psis.item()))
        self.acmi_writer.write(logmsg("B01","F16","Blue",blue_pos_y.item() + 10.0, blue_pos_x.item() + 10.0, blue_pos_z.item(), blue_miu.item(), blue_gammas.item(), blue_psis.item()))
        self.acmi_writer.write(logmsg("C01","F16","Green",obstacle_1_y.item()+10.0,obstacle_1_x.item()+10.0,obstacle_1_z.item(),obstacle_1_miu.item(),obstacle_1_gammas.item(),obstacle_1_psis.item()))
        self.acmi_writer.write(logmsg("C02","F16","Green",obstacle_2_y.item()+10.0,obstacle_2_x.item()+10.0,obstacle_2_z.item(),obstacle_2_miu.item(),obstacle_2_gammas.item(),obstacle_2_psis.item()))
        self.acmi_writer.write(logmsg("C03","F16","Green",obstacle_3_y.item()+10.0,obstacle_3_x.item()+10.0,obstacle_3_z.item(),obstacle_3_miu.item(),obstacle_3_gammas.item(),obstacle_3_psis.item()))
        self.acmi_writer.write(logmsg("C04","F16","Green",obstacle_4_y.item()+10.0,obstacle_4_x.item()+10.0,obstacle_4_z.item(),obstacle_4_miu.item(),obstacle_4_gammas.item(),obstacle_4_psis.item()))
        self.acmi_writer.write(logmsg("C05","F16","Green",obstacle_5_y.item()+10.0,obstacle_5_x.item()+10.0,obstacle_5_z.item(),obstacle_5_miu.item(),obstacle_5_gammas.item(),obstacle_5_psis.item()))

        self.safe_writer.write(f"#{self.time:.2f}\n")
        for i in range(5):
            distance_obstacle = torch.sqrt((pos[0:1, 0:1, 2:3]-pos[0:1,i+2:i+3,2:3])*(pos[0:1, 0:1, 2:3]-pos[0:1,i+2:i+3,2:3])+(pos[0:1, 0:1, 0:1]-pos[0:1,i+2:i+3,0:1])*(pos[0:1, 0:1, 0:1]-pos[0:1,i+2:i+3,0:1])+(pos[0:1, 0:1, 1:2]-pos[0:1,i+2:i+3,1:2])*(pos[0:1, 0:1, 1:2]-pos[0:1,i+2:i+3,1:2]))
            self.safe_writer.write(str((distance_obstacle).cpu().numpy().item())+"\n")

        self.time += 1

        # ship inx
        ship_inx = self.num_actors 

        # set v to isaac gym
        blue_root_linvels = self.root_states[:, ship_inx:ship_inx+1, 7:10]
        blue_copy_v = - blue_root_linvels.clone()

        root_linvels = self.root_states[:, :self.num_actors, 7:10]
        red_copy_v = - root_linvels.clone()

        self.root_states[:, :self.num_actors, 7:10] += red_v + red_copy_v
        self.root_states[:, ship_inx:ship_inx+1, 7:10] += (blue_v + blue_copy_v)

        # reset ids
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0 and reset_env_ids[0] == 0:
            self.flog += 1
            self.time = 0
            self.acmi_writer,self.safe_writer = set_excel(self.flog)

        if len(reset_env_ids) > 0:

            actor_indices = self.all_actor_indices[reset_env_ids].flatten()

            num_resets = len(reset_env_ids)
            self.root_states[reset_env_ids] = self.initial_root_states[reset_env_ids]

            for i in range(self.num_actors):
                self.red_uav.reset(reset_env_ids, num_resets, i)
                self.blue_uav.reset(reset_env_ids, num_resets, i)

                self.root_states[reset_env_ids, i, 0:3] += self.red_uav.pos[reset_env_ids, i, 0:3]
                self.root_states[reset_env_ids, 1:2, 0:3] += self.blue_uav.pos[reset_env_ids, i:i+1, 0:3]

            self.reset_buf[reset_env_ids] = 0
            self.progress_buf[reset_env_ids] = 0

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)

    def post_physics_step(self):

        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

    def compute_observations(self):
        quadcopter_pos = torch.squeeze(self.root_states[:, :self.num_actors, 0:3])
        ship_pos = self.rigid_positions[0]

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        y_axis = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)

        pu_pt = torch.squeeze(self.root_states[:, 1, 0:3] - self.root_states[:, 0, 0:3])
        pu_pt_oxy = torch.squeeze(self.root_states[:, 1, 0:2] - self.root_states[:, 0, 0:2])

        theta_u = get_angle_between_vectors(torch.squeeze(self.root_states[:, 0, 7:10]), pu_pt)
        theta_t = get_angle_between_vectors(torch.squeeze(self.root_states[:, 1, 7:10]), pu_pt)

        dis_red_and_blue = torch.sqrt(
            (ship_pos[..., 0] - quadcopter_pos[..., 0]) * (ship_pos[..., 0] - quadcopter_pos[..., 0])
            + (ship_pos[..., 1] - quadcopter_pos[..., 1]) * (ship_pos[..., 1] - quadcopter_pos[..., 1])
            + (ship_pos[..., 2] - quadcopter_pos[..., 2]) * (ship_pos[..., 2] - quadcopter_pos[..., 2])
        )

        z_u = torch.squeeze(self.root_states[:, 0, 2])
        z_t = torch.squeeze(self.root_states[:, 1, 2])

        # add the position of the obstacle
        own_uav_x = torch.squeeze(self.root_states[:,0,0])
        own_uav_y = torch.squeeze(self.root_states[:,0,1])
        own_uav_z = torch.squeeze(self.root_states[:,0,2])

        # add the position of the obstacle
        distance_red_obs = torch.zeros(self.num_object-1,self.num_envs,dtype=torch.float32,device=self.device)
        for j in range(self.num_object-1):
            obstacle_x = torch.squeeze(self.root_states[:,j+2,0])
            obstacle_y = torch.squeeze(self.root_states[:,j+2,1])
            obstacle_z = torch.squeeze(self.root_states[:,j+2,2])
            # add the distance between red and obstacle
            distance_red_obs[j] = torch.sqrt((own_uav_x-obstacle_x)*(own_uav_x-obstacle_x)+(own_uav_y-obstacle_y)*(own_uav_y-obstacle_y)+(own_uav_z-obstacle_z)*(own_uav_z-obstacle_z)) 
            print(distance_red_obs[j][0])
        a, b = 10, 5
        for i in range(self.num_actors):
            idx = i * self.num_obs_per_actor

            # The normalized 13-dimensional state space.
            self.obs_buf[..., idx + 0] = torch.squeeze(self.red_uav.v) / 4.0 * a - b
            self.obs_buf[..., idx + 1] = torch.squeeze(self.red_uav.gammas) / (2.0 * math.pi) * a - b
            self.obs_buf[..., idx + 2] = torch.squeeze(self.red_uav.psis) / (2.0 * math.pi) * a - b
            self.obs_buf[..., idx + 3] = torch.squeeze((self.red_uav.v - self.blue_uav.v) / 3.5) * a - b
            self.obs_buf[..., idx + 4] = torch.squeeze(self.blue_uav.gammas) / (2.0 * math.pi) * a - b
            self.obs_buf[..., idx + 5] = torch.squeeze(self.blue_uav.psis) / (2.0 * math.pi) * a - b
            self.obs_buf[..., idx + 6] = dis_red_and_blue / 100.0 * a - b
            self.obs_buf[..., idx + 7] = 2 * b * (90.0 - get_angle_between_vectors(pu_pt, z_axis)) / 180.0

            pp = get_angle_between_vectors(pu_pt_oxy, y_axis)
            psi_r = torch.where(pu_pt_oxy[..., 0] >= 0.0, pp, 360.0 - pp)
            self.obs_buf[..., idx + 8] = psi_r / 360.0

            self.obs_buf[..., idx + 9] = theta_u / 360.0 * a - b
            self.obs_buf[..., idx + 10] = theta_t / 360.0 * a - b
            self.obs_buf[..., idx + 11] = z_u / 120.0 * a - b
            self.obs_buf[..., idx + 12] = (z_u - z_t) / (120.0 - 10.0) * a - b

            # self.obs_buf[...,idx+13] = own_uav_x/100 * a -b
            # self.obs_buf[...,idx+14] = own_uav_y/100 * a -b
            # self.obs_buf[...,idx+15] = own_uav_z/100 * a -b
            # self.obs_buf[...,idx+16] = obstacle_x/100 * a -b
            # self.obs_buf[...,idx+17] = obstacle_y/100 * a -b
            # self.obs_buf[...,idx+18] = obstacle_z/100 * a -b

            self.obs_buf[...,idx+13] = distance_red_obs[0]/1000.0*a-b
            self.obs_buf[...,idx+14] = distance_red_obs[1]/1000.0*a-b
            self.obs_buf[...,idx+15] = distance_red_obs[2]/1000.0*a-b
            self.obs_buf[...,idx+16] = distance_red_obs[3]/1000.0*a-b
            self.obs_buf[...,idx+17] = distance_red_obs[4]/1000.0*a-b

            

        r = copy.deepcopy(self.red_uav)
        b = copy.deepcopy(self.blue_uav)
        self.blue_uav.set_states(b, r, self.root_states, dis_red_and_blue)
        blue_state = self.blue_uav.get_states()
        # self.blue_obs_buf[..., 0:13] = blue_state

        self.miu_reward = 1.0 - (theta_u + theta_t) / 180.0
        self.blue_miu_reward = 1.0 - (360.0 - (theta_u + theta_t)) / 180.0

        
        # now , we calculate the value to select the policy
        # first , if it is going to collisode , collision = 1,else = 0
        # and , collision_value = 2*collisode - 1, so it is  1 ,else ,it is -1 ; 
        collision = torch.zeros_like(self.reset_buf)
        ones = torch.ones_like(self.reset_buf)
        nagtive = torch.ones_like(self.reset_buf)*(-1)
        # nearest_distance = torch.ones_like(self.reset_buf)*10

        for j in range(self.num_object-1):
            obstacle_x = torch.squeeze(self.root_states[:,j+2,0])
            obstacle_y = torch.squeeze(self.root_states[:,j+2,1])
            obstacle_z = torch.squeeze(self.root_states[:,j+2,2])
            # add the distance between red and obstacle
            distance_red_obs[j] = torch.sqrt((own_uav_x-obstacle_x)*(own_uav_x-obstacle_x)+(own_uav_y-obstacle_y)*(own_uav_y-obstacle_y)+(own_uav_z-obstacle_z)*(own_uav_z-obstacle_z)) 
            # print(distance_red_obs[j])
        nearest_distance , _ = torch.min(distance_red_obs,dim=0)
        
        collision_value = torch.where(nearest_distance<15,(ones)*((15-nearest_distance)/15),nagtive)
        # collision_value = torch.where(nearest_distance<15.0,ones,nagtive)
        # print(collision_value)

        distance_value = (-1)*torch.tanh(torch.log(torch.clamp(dis_red_and_blue/3,min=1e-10)))

        # if safe_value is positive , it means too close to the obstacle , we should choose recovery pilicy
        # else if safe value is negative , it means it si normal , and we choose reward policy
        # safe_value = distance_value  + collision_value
        # print(safe_value)
        safe_value = torch.max(distance_value , collision_value)
        # print(safe_value)
        self.safe_values_buf = safe_value


    def compute_reward(self):

        p_unsafe = torch.zeros_like(self.reset_buf, dtype=torch.float32)

        pos_reward = torch.zeros_like(self.reset_buf, dtype=torch.float32)


        self.safe_reward = pos_reward + p_unsafe
        self.safe_reward_buf = self.safe_reward