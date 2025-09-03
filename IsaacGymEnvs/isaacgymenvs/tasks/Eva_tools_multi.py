import copy
import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
import random
from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask


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

def pursuit_run_pos_func(self, target):
    p_t = self.pos
    p_u = target.pos
    pu_pt = torch.squeeze(p_t - p_u)
    # pu_pt = torch.squeeze(p_u - p_t)

    self_v = torch.squeeze(self.vv)
    target_v = torch.squeeze(target.vv)

    theta_u = get_angle_between_vectors(self_v, pu_pt)
    theta_t = get_angle_between_vectors(target_v, pu_pt)

    return 1.0 - (theta_t + theta_u) / 180.0

def pursuit_attack_pos_func(self, target):
    p_t = self.pos
    p_u = target.pos
    pu_pt = torch.squeeze(p_u - p_t)
    # pu_pt = torch.squeeze(p_u - p_t)

    self_v = torch.squeeze(self.vv)
    target_v = torch.squeeze(target.vv)

    theta_u = get_angle_between_vectors(self_v, pu_pt)
    theta_t = get_angle_between_vectors(target_v, pu_pt)

    return 1.0 - (theta_t + theta_u) / 180.0

def pursuit_pos_func(self, target):
    p_t = self.pos
    p_u = target.pos
    pu_pt = torch.squeeze(p_t - p_u)
    # pu_pt = torch.squeeze(p_u - p_t)

    self_v = torch.squeeze(self.vv)
    target_v = torch.squeeze(target.vv)

    theta_u = get_angle_between_vectors(self_v, pu_pt)
    theta_t = get_angle_between_vectors(target_v, pu_pt)

    return 1.0 - (theta_t + theta_u) / 180.0


def pursuit_dis_func(self, target):
    p_t = self.pos
    p_u = target.pos
    p = torch.squeeze(p_t - p_u)
    dist = torch.norm(p, dim=1)

    return 1 / (1 + (dist / 10.0) ** 2)


def constraint_dis(self, target):
    p_t = self.pos
    p_u = target.pos
    p = torch.squeeze(p_t - p_u)
    dist = torch.norm(p, dim=1)
    punish = torch.where(dist <= 3.0, - torch.ones_like(dist), torch.zeros_like(dist))

    return punish


class Uavs_posture:
    def __init__(self, quaternion, num_env, device):
        self.quaternion = quaternion.squeeze(dim=1)
        self.num_env = num_env
        self.device = device

        self.dg = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 1)
        self.dp = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 1)
        self._miu = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 1)
        self.dmiu = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 1)

    # tong guo quaternion xuan zhuan vector
    def rotate_voctor_with_quaternion(self, vector, quaternion):
        norm = torch.linalg.norm(quaternion)
        quaternion = quaternion / norm

        vec_quat = torch.zeros(self.num_env, 4).to(vector.device)
        vec_quat[:, :3] = vector

        quat_conj = quaternion.clone()
        quat_conj[:, :3] *= -1

        temp = self.quaternion_mult(quaternion, vec_quat)
        rotated_vec_quat = self.quaternion_mult(temp, quat_conj)

        return rotated_vec_quat[:, :3]

    def quaternion_mult(self, q1, q2):
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.cat([x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1), w.unsqueeze(dim=1)], dim=1).to(q1.device)

    def rotate_to_horizontal(self, plane_quaternion, device):
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float)
        x = self.rotate_voctor_with_quaternion(x_axis, plane_quaternion)

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


class Uavs_red(Uavs_posture):
    def __init__(self, pos, v, gamma, psi, dt, num_env, device, quaternion):
        super().__init__(quaternion, num_env, device)
        self.nx, self.nz = 0, 0
        self.miu = torch.tensor([0.0], device=self.device, dtype=torch.float32).repeat(num_env, 1, 1)
        self.dt = dt
        self.g = 9.81
        self.device = device
        self.num_env = num_env

        self.pos = pos

        self.v = v
        self.v = torch.clamp(self.v, min=0.5, max=4.0)
        self.vv = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 3)

        self.gammas = gamma
        self.psis = psi

        self.min_v = 0.5
        self.max_v = 4.0

    def move(self, action):
        self.v = torch.clamp(self.v, min=self.min_v, max=self.max_v)

        self.nx = action[:, 0].unsqueeze(dim=1).unsqueeze(dim=1)
        self.nz = action[:, 1].unsqueeze(dim=1).unsqueeze(dim=1)
        self.miu = action[:, 2].unsqueeze(dim=1).unsqueeze(dim=1)

        dv = self.g * (self.nx - torch.sin(self.gammas))
        dg = self.g / (self.v * 100.0) * (self.nz * torch.cos(self.miu) - torch.cos(self.gammas))
        dp = self.g * self.nz * torch.sin(self.miu) / ((self.v * 100.0) * torch.cos(self.gammas))

        self.dg = dg
        self.dp = dp

        self.v += dv * self.dt / 100.0
        self.gammas += dg * self.dt
        self.psis += dp * self.dt

        v_x = self.v * torch.cos(self.gammas) * torch.sin(self.psis)
        v_y = self.v * torch.cos(self.gammas) * torch.cos(self.psis)
        v_z = self.v * torch.sin(self.gammas)

        self.vv = torch.cat((v_x, v_y, v_z), dim=2)

    def update_pos(self, pos):
        self.pos = pos

    def reset(self, reset_ids, num_resets, i):
        self.v[reset_ids, i] = torch.tensor([2.5], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
        self.gammas[reset_ids, i] = (random.uniform(-math.pi/6.0, math.pi/6.0) *
                                     torch.ones(1, device=self.device, dtype=torch.float32).repeat(num_resets, 1))
        self.psis[reset_ids, i] = (random.uniform(-math.pi, math.pi) *
                                   torch.ones(1, device=self.device, dtype=torch.float32).repeat(num_resets, 1))

        self.pos[reset_ids, i, 0:3] = 0.0
        self.pos[reset_ids, i, 0] += torch_rand_float(-10.0, 10.0, (num_resets, 1), self.device).flatten()
        self.pos[reset_ids, i, 1] += torch_rand_float(-30.0, -20.0, (num_resets, 1), self.device).flatten()
        self.pos[reset_ids, i, 2] += torch_rand_float(45.0, 55.0, (num_resets, 1), self.device).flatten()

    def update_angle_v(self, quaternion):
        self.dmiu = torch.where((self._miu - self.miu) < 0, (self._miu - self.miu) * torch.ones_like(self.miu), torch.zeros_like(self.miu))
        self.dmiu = torch.where((self._miu - self.miu) > 0, - (self._miu - self.miu) * torch.ones_like(self.miu), torch.zeros_like(self.miu))

        self._miu = self.miu
        self.quaternion = quaternion.squeeze(dim=1)

    def get_angle_v(self, quaternion):
        self.update_angle_v(quaternion)

        dg = self.dg.squeeze(dim=1)
        dg = torch.cat((dg, torch.zeros(self.num_env, 1, device=self.device), torch.zeros(self.num_env, 1, device=self.device)), dim=1)

        dp = self.dp.squeeze(dim=1)
        dp = torch.cat((torch.zeros(self.num_env, 1, device=self.device), torch.zeros(self.num_env, 1, device=self.device), dp), dim=1)

        dm = self.dmiu.squeeze(dim=1)
        dm = torch.cat((torch.zeros(self.num_env, 1, device=self.device), dm, torch.zeros(self.num_env, 1, device=self.device)), dim=1)

        g_v = self.rotate_voctor_with_quaternion(dg, self.quaternion)
        p_v = self.rotate_voctor_with_quaternion(- dp, self.quaternion)
        m_v = self.rotate_voctor_with_quaternion(dm, self.quaternion)

        return (g_v + p_v + m_v).unsqueeze(dim=1)


class Uavs_blue(Uavs_posture):
    def __init__(self, pos, v, gamma, psi, dt, num_env, device, quaternion):
        super().__init__(quaternion, num_env, device)
        self.nx, self.nz, self.miu = 0, 0, torch.tensor([0.0], device=device, dtype=torch.float32).repeat(num_env, 1, 1)
        self.dt = dt
        self.g = 9.81
        self.device = device
        self.num_env = num_env

        self.pos = pos

        self.v = v
        self.v = torch.clamp(self.v, min=0.5, max=4.0)
        self.vv = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 3)

        self.gammas = gamma
        self.psis = psi

        self.min_v = 0.5
        self.max_v = 4.0

        self.action_space = torch.tensor([[0.0, 4.0, -math.acos(1.0 / 4.0)], [1.5, 1.0, 0.0], [0.0, 1.0, 0.0],
                                          [-1.0, 1.0, 0.0], [0.0, 4.0, math.acos(1.0 / 4.0)], [0.0, 4.0, 0.0],
                                          [0.0, 4.0, math.pi], [2.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
                                          [0.0, 8.0, math.acos(1.0 / 8.0)], [2.0, 8.0, math.acos(1.0 / 8.0)], [-1.0, 8.0, math.acos(1.0 / 8.0)],
                                          [0.0, 8.0, - math.acos(1.0 / 8.0)], [2.0, 8.0, -math.acos(1.0 / 8.0)], [-1.0, 8.0, -math.acos(1.0 / 8.0)],
                                        ],
                                         device=self.device, dtype=torch.float32).repeat(self.num_env, 1, 1)

        self.states = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(self.num_env, 13)

    def move(self, action):
        self.v = torch.clamp(self.v, min=self.min_v, max=self.max_v)

        self.nx = action[:, 0].unsqueeze(dim=1).unsqueeze(dim=1)
        self.nz = action[:, 1].unsqueeze(dim=1).unsqueeze(dim=1)
        self.miu = action[:, 2].unsqueeze(dim=1).unsqueeze(dim=1)

        dv = self.g * (self.nx - torch.sin(self.gammas))
        dg = self.g / (self.v * 100.0) * (self.nz * torch.cos(self.miu) - torch.cos(self.gammas))
        dp = self.g * self.nz * torch.sin(self.miu) / ((self.v * 100.0) * torch.cos(self.gammas))

        self.dg = dg
        self.dp = dp

        self.v += dv * self.dt / 100.0
        self.gammas += dg * self.dt
        self.psis += dp * self.dt

        v_x = self.v * torch.cos(self.gammas) * torch.sin(self.psis)
        v_y = self.v * torch.cos(self.gammas) * torch.cos(self.psis)
        v_z = self.v * torch.sin(self.gammas)

        self.vv = torch.cat((v_x, v_y, v_z), dim=2)

        self.pos[:, 0, 0] += torch.squeeze(v_x) * self.dt
        self.pos[:, 0, 1] += torch.squeeze(v_y) * self.dt
        self.pos[:, 0, 2] += torch.squeeze(v_z) * self.dt

    def update_pos(self, pos):
        self.pos = pos

    def select_policy(self, red, blue, index):
        size_matrix = 15

        table = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(size_matrix, size_matrix, self.num_env)
        for i in range(size_matrix):
            for j in range(size_matrix):
                r = copy.deepcopy(red)
                b = copy.deepcopy(blue)
                table[i, j, :] = torch.where(torch.squeeze(index) == 1, self.get_attack_score(b, r, i, j), self.get_run_score(b, r, i, j))
        column_min, _ = torch.min(table, dim=1)
        _, max_b = torch.max(column_min, dim=0)
        aaa = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(15, self.num_env)
        max_b = max_b.unsqueeze(-1)

        action = self.action_space[0, max_b]
        action = torch.squeeze(action)

        return action

    def select_better_action(self, red, blue, red_actions):
        flag = torch.tensor([2.0], device=self.device, dtype=torch.float32).repeat(self.num_env)
        ones = torch.ones_like(flag)
        zeros = torch.zeros_like(flag)
        size_matrix = 3

        table = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(size_matrix, self.num_env)
        for i in range(size_matrix):
            r = copy.deepcopy(red)
            b = copy.deepcopy(blue)
            table[i, :] = self.get_better_score(b, r, i, red_actions)
        _, max_b = torch.max(table, dim=0)
        max_b = max_b.unsqueeze(-1)

        action = self.action_space[0, max_b]
        action = torch.squeeze(action)

        return action

    def select_action(self, red, blue):
        flag = torch.tensor([2.0], device=self.device, dtype=torch.float32).repeat(self.num_env)
        ones = torch.ones_like(flag)
        zeros = torch.zeros_like(flag)
        size_matrix = 15

        table = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(size_matrix, size_matrix, self.num_env)
        for i in range(size_matrix):
            for j in range(size_matrix):
                r = copy.deepcopy(red)
                b = copy.deepcopy(blue)
                table[i, j, :] = self.get_score(b, r, i, j)
        column_min, _ = torch.min(table, dim=1)
        _, max_b = torch.max(column_min, dim=0)
        aaa = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(15, self.num_env)
        max_b = max_b.unsqueeze(-1)

        action = self.action_space[0, max_b]
        action = torch.squeeze(action)

        b = copy.deepcopy(blue)
        b.move(action)

        flag = torch.where(b.pos[:, 0, 2] > 120.0, ones, flag)
        flag = torch.where(b.pos[:, 0, 2] < 10.0, zeros, flag)

        action = torch.where(flag.unsqueeze(-1) == 0, torch.tensor([0.0, 8.0, 0.0], device=self.device, dtype=torch.float32).repeat(self.num_env, 1), action)
        action = torch.where(flag.unsqueeze(-1) == 1, torch.tensor([0.0, 8.0, math.pi], device=self.device, dtype=torch.float32).repeat(self.num_env, 1), action)

        gammas = torch.where(flag.unsqueeze(-1) == 0, torch.zeros_like(self.gammas).squeeze(dim=1), self.gammas.squeeze(dim=1))
        miu = torch.where(flag.unsqueeze(-1) == 0, torch.zeros_like(self.miu).squeeze(dim=1), self.miu.squeeze(dim=1))

        gammas = torch.where(flag.unsqueeze(-1) == 1, torch.zeros_like(self.gammas).squeeze(dim=1), self.gammas.squeeze(dim=1))
        miu = torch.where(flag.unsqueeze(-1) == 1, torch.zeros_like(self.miu).squeeze(dim=1), self.miu.squeeze(dim=1))

        self.gammas = gammas.unsqueeze(dim=1)
        self.miu = miu.unsqueeze(dim=1)

        return action

    def choose_action(self, index):
        action = self.action_space[0, index.long()]
        action = torch.squeeze(action)

        return action

    def get_real_move(self, blue, action):
        flag = torch.tensor([2.0], device=self.device, dtype=torch.float32).repeat(self.num_env)
        ones = torch.ones_like(flag)
        zeros = torch.zeros_like(flag)

        b = copy.deepcopy(blue)
        b.move(action)

        flag = torch.where(b.pos[:, 0, 2] > 116.0, ones, flag)
        flag = torch.where(b.pos[:, 0, 2] < 14.0, zeros, flag)

        action = torch.where(flag.unsqueeze(-1) == 0,
                             torch.tensor([0.0, 8.0, 0.0], device=self.device, dtype=torch.float32).repeat(self.num_env,
                                                                                                           1), action)
        action = torch.where(flag.unsqueeze(-1) == 1,
                             torch.tensor([0.0, 8.0, math.pi], device=self.device, dtype=torch.float32).repeat(
                                 self.num_env, 1), action)

        gammas = torch.where(flag.unsqueeze(-1) == 0, torch.zeros_like(self.gammas).squeeze(dim=1),
                             self.gammas.squeeze(dim=1))
        miu = torch.where(flag.unsqueeze(-1) == 0, torch.zeros_like(self.miu).squeeze(dim=1), self.miu.squeeze(dim=1))

        gammas = torch.where(flag.unsqueeze(-1) == 1, torch.zeros_like(self.gammas).squeeze(dim=1),
                             self.gammas.squeeze(dim=1))
        miu = torch.where(flag.unsqueeze(-1) == 1, torch.zeros_like(self.miu).squeeze(dim=1), self.miu.squeeze(dim=1))

        self.gammas = gammas.unsqueeze(dim=1)
        self.miu = miu.unsqueeze(dim=1)

        return action

    def get_better_score(self, blue, red, b, r):
        red.move(r)
        blue.move(self.action_space[:, b])
        p = pursuit_pos_func(blue, red)
        d = pursuit_dis_func(blue, red)
        score = p + p * d
        return score

    def get_score(self, blue, red, b, r):
        red.move(self.action_space[:, r])
        blue.move(self.action_space[:, b])
        p = pursuit_pos_func(blue, red)
        d = pursuit_dis_func(blue, red)
        punish = constraint_dis(blue, red)
        score = p + d * p
        # score = d + d * p + punish
        return score

    def get_attack_score(self, blue, red, b, r):
        red.move(self.action_space[:, r])
        blue.move(self.action_space[:, b])
        p = pursuit_attack_pos_func(blue, red)
        d = pursuit_dis_func(blue, red)
        punish = constraint_dis(blue, red)
        score = d + d * p + punish
        return score

    def get_run_score(self, blue, red, b, r):
        red.move(self.action_space[:, r])
        blue.move(self.action_space[:, b])
        p = pursuit_run_pos_func(blue, red)
        d = pursuit_dis_func(blue, red)
        score = p + d * p
        return score

    def reset(self, reset_ids, num_resets, i):
        self.v[reset_ids, i] = torch.tensor([1.5], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
        self.gammas[reset_ids, i] = torch.zeros(1, device=self.device, dtype=torch.float32).repeat(num_resets, 1)
        self.psis[reset_ids, i] = (random.uniform(-math.pi, math.pi) *
                                   torch.ones(1, device=self.device, dtype=torch.float32).repeat(num_resets, 1))

        self.pos[reset_ids, i, 0:3] = 0.0
        self.pos[reset_ids, i, 2] += torch_rand_float(40.0, 60.0, (num_resets, 1), self.device).flatten()

    def set_states(self, red_uav, blue_uav, root_states, dis_red_and_blue):
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float).repeat(self.num_env, 1)
        y_axis = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float).repeat(self.num_env, 1)

        pu_pt = torch.squeeze(root_states[:, 0, 0:3] - root_states[:, 1, 0:3])
        pu_pt_oxy = torch.squeeze(root_states[:, 0, 0:2] - root_states[:, 1, 0:2])

        theta_u = get_angle_between_vectors(torch.squeeze(root_states[:, 1, 7:10]), pu_pt)
        theta_t = get_angle_between_vectors(torch.squeeze(root_states[:, 0, 7:10]), pu_pt)

        z_u = torch.squeeze(root_states[:, 1, 2])
        z_t = torch.squeeze(root_states[:, 0, 2])

        a, b = 10, 5
        self.states[..., 0] = torch.squeeze(red_uav.v) / 4.0 * a - b
        self.states[..., 1] = torch.squeeze(red_uav.gammas) / (2.0 * math.pi) * a - b
        self.states[..., 2] = torch.squeeze(red_uav.psis) / (2.0 * math.pi) * a - b
        self.states[..., 3] = torch.squeeze(- (red_uav.v - blue_uav.v) / 3.5) * a - b
        self.states[..., 4] = torch.squeeze(blue_uav.gammas) / (2.0 * math.pi) * a - b
        self.states[..., 5] = torch.squeeze(blue_uav.psis) / (2.0 * math.pi) * a - b
        self.states[..., 6] = dis_red_and_blue / 100.0 * a - b
        self.states[..., 7] = 2 * b * (90.0 - get_angle_between_vectors(pu_pt, z_axis)) / 180.0

        pp = get_angle_between_vectors(pu_pt_oxy, y_axis)
        psi_r = torch.where(pu_pt_oxy[..., 0] >= 0.0, pp, 360.0 - pp)
        self.states[..., 8] = psi_r / 360.0

        self.states[..., 9] = theta_u / 360.0 * a - b
        self.states[..., 10] = theta_t / 360.0 * a - b
        self.states[..., 11] = z_u / 120.0 * a - b
        self.states[..., 12] = (z_u - z_t) / (120.0 - 10.0) * a - b

    def get_states(self):
        return self.states

    def update_angle_v(self, quaternion):
        self.dmiu = torch.where((self._miu - self.miu) < 0, (self._miu - self.miu) * torch.ones_like(self.miu), torch.zeros_like(self.miu))
        self.dmiu = torch.where((self._miu - self.miu) > 0, - (self._miu - self.miu) * torch.ones_like(self.miu), torch.zeros_like(self.miu))

        self._miu = self.miu
        self.quaternion = quaternion.squeeze(dim=1)

    def get_angle_v(self, quaternion):
        self.update_angle_v(quaternion)

        dg = self.dg.squeeze(dim=1)
        dg = torch.cat((dg, torch.zeros(self.num_env, 1, device=self.device), torch.zeros(self.num_env, 1, device=self.device)), dim=1)

        dp = self.dp.squeeze(dim=1)
        dp = torch.cat((torch.zeros(self.num_env, 1, device=self.device), torch.zeros(self.num_env, 1, device=self.device), dp), dim=1)

        dm = self.dmiu.squeeze(dim=1)
        dm = torch.cat((torch.zeros(self.num_env, 1, device=self.device), dm, torch.zeros(self.num_env, 1, device=self.device)), dim=1)

        # g_v = self.rotate_voctor_with_quaternion(dg, self.quaternion)
        # p_v = self.rotate_voctor_with_quaternion(- dp, self.quaternion)
        # m_v = self.rotate_voctor_with_quaternion(dm, self.quaternion)

        g_v = quat_rotate(self.quaternion, dg)
        p_v = quat_rotate(self.quaternion, - dp)
        m_v = quat_rotate(self.quaternion, dm)

        return (g_v + p_v + m_v).unsqueeze(dim=1)
