from typing import Tuple

import jax
import jax.numpy as jnp

from ...env.lidar_env.base import LidarEnv
from ...utils.graph import GraphsTuple
from ...utils.typing import Action, AgentState, Array


def _agent_position(state: Array) -> Array:
    return state[:2]


def _agent_velocity(state: Array) -> Array:
    if state.shape[0] == 5:
        return state[4] * state[2:4]
    return state[2:4]


def _entity_drift(entity_states: AgentState, state_dim: int) -> AgentState:
    # 邻居 agent 被视为会运动的实体；它们的运动不由当前 agent 控制，
    # 因此会进入局部约束漂移项 psi。后面的 lidar hit 点则使用零漂移。
    if state_dim == 5:
        speed = entity_states[:, 4]
        drift = jnp.stack(
            [
                speed * entity_states[:, 2],
                speed * entity_states[:, 3],
                jnp.zeros_like(speed),
                jnp.zeros_like(speed),
                jnp.zeros_like(speed),
            ],
            axis=1,
        )
    else:
        drift = jnp.concatenate(
            [entity_states[:, 2:4], jnp.zeros((entity_states.shape[0], 2), dtype=entity_states.dtype)],
            axis=1,
        )
    return drift


def _pairwise_collision_h(
        own_state: Array,
        entity_states: AgentState,
        active: Array,
        safe_radius: Array,
        braking_accel: float,
        velocity_margin: float,
) -> Array:
    # HMM 风格的两两安全约束：
    #   h = (r_safe + braking_margin)^2 - ||p_i - p_j||^2 <= 0
    # braking_margin 让 h 显式依赖速度。对 LidarEnv 的二阶积分器来说，
    # 如果只用位置距离约束，action 是加速度，一阶上不会直接影响 h_dot。
    own_pos = _agent_position(own_state)
    own_vel = _agent_velocity(own_state)
    entity_pos = entity_states[:, :2]
    if own_state.shape[0] == 5:
        entity_vel = entity_states[:, 4:5] * entity_states[:, 2:4]
    else:
        entity_vel = entity_states[:, 2:4]

    rel_pos = own_pos[None, :] - entity_pos
    rel_vel = own_vel[None, :] - entity_vel
    dist = jnp.linalg.norm(rel_pos, axis=-1)
    direction = rel_pos / jnp.maximum(dist[:, None], 1e-6)
    approach_speed = jnp.maximum(0.0, -jnp.sum(rel_vel * direction, axis=-1))
    braking_margin = velocity_margin + approach_speed ** 2 / (2.0 * braking_accel)
    h = (safe_radius + braking_margin) ** 2 - dist ** 2
    # inactive 的 padding/truncated 约束保持为安全的负值，避免触发动作修正。
    return jnp.where(active, h, -1.0)


def _velocity_state_jac(state: Array) -> Array:
    if state.shape[0] == 5:
        speed = state[4]
        cos_theta = state[2]
        sin_theta = state[3]
        jac = jnp.zeros((2, state.shape[0]), dtype=state.dtype)
        jac = jac.at[0, 2].set(speed)
        jac = jac.at[1, 3].set(speed)
        jac = jac.at[:, 4].set(jnp.array([cos_theta, sin_theta], dtype=state.dtype))
        return jac
    jac = jnp.zeros((2, state.shape[0]), dtype=state.dtype)
    return jac.at[:, 2:4].set(jnp.eye(2, dtype=state.dtype))


def _pairwise_collision_h_jac(
        own_state: Array,
        entity_states: AgentState,
        active: Array,
        safe_radius: Array,
        braking_accel: float,
        velocity_margin: float,
) -> Tuple[Array, Array, Array]:
    # 解析计算 h 及其对 own/entity state 的雅可比，避免在 rollout
    # 内部反复调用 jacfwd。符号约定与 _pairwise_collision_h 一致：
    # r = p_own - p_entity, v = v_own - v_entity。
    own_pos = _agent_position(own_state)
    own_vel = _agent_velocity(own_state)
    entity_pos = entity_states[:, :2]
    if own_state.shape[0] == 5:
        entity_vel = entity_states[:, 4:5] * entity_states[:, 2:4]
    else:
        entity_vel = entity_states[:, 2:4]

    rel_pos = own_pos[None, :] - entity_pos
    rel_vel = own_vel[None, :] - entity_vel
    dist = jnp.linalg.norm(rel_pos, axis=-1)
    safe_dist = jnp.maximum(dist, 1e-6)
    direction = rel_pos / safe_dist[:, None]
    closing_raw = -jnp.sum(rel_vel * direction, axis=-1)
    is_closing = closing_raw > 0.0
    approach_speed = jnp.where(is_closing, closing_raw, 0.0)
    braking_margin = velocity_margin + approach_speed ** 2 / (2.0 * braking_accel)
    radius_with_margin = safe_radius + braking_margin
    h = radius_with_margin ** 2 - dist ** 2

    eye2 = jnp.eye(2, dtype=own_state.dtype)
    dir_jac = (eye2[None, :, :] - direction[:, :, None] * direction[:, None, :]) / safe_dist[:, None, None]
    dclosing_drel_pos = -jnp.einsum("mij,mj->mi", dir_jac, rel_vel)
    dapproach_scale = jnp.where(is_closing, approach_speed / braking_accel, 0.0)
    margin_scale = 2.0 * radius_with_margin * dapproach_scale

    dh_drel_pos = margin_scale[:, None] * dclosing_drel_pos - 2.0 * rel_pos
    dh_down_vel = margin_scale[:, None] * (-direction)
    dh_dentity_vel = margin_scale[:, None] * direction

    own_vel_jac = _velocity_state_jac(own_state)
    entity_vel_jac = jax.vmap(_velocity_state_jac)(entity_states)
    dh_down = jnp.zeros((entity_states.shape[0], own_state.shape[0]), dtype=own_state.dtype)
    dh_down = dh_down.at[:, :2].set(dh_drel_pos)
    dh_down = dh_down + dh_down_vel @ own_vel_jac

    dh_dentities = jnp.zeros_like(entity_states)
    dh_dentities = dh_dentities.at[:, :2].set(-dh_drel_pos)
    dh_dentities = dh_dentities + jnp.einsum("mi,mij->mj", dh_dentity_vel, entity_vel_jac)

    h = jnp.where(active, h, -1.0)
    dh_down = jnp.where(active[:, None], dh_down, 0.0)
    dh_dentities = jnp.where(active[:, None], dh_dentities, 0.0)
    return h, dh_down, dh_dentities


def _single_agent_project(
        action_dim: int,
        dt: float,
        own_state: Array,
        own_drift: Array,
        own_control_matrix: Array,
        entity_states: AgentState,
        entity_drift: AgentState,
        active: Array,
        safe_radius: Array,
        mu: Array,
        u_ref: Array,
        braking_accel: float,
        velocity_margin: float,
        contraction_gain: float,
        slack_min: float,
        slack_beta: float,
        slack_weight: float,
        reg: float,
) -> Tuple[Array, Array]:
    h, dh_down, dh_dentities = _pairwise_collision_h_jac(
        own_state,
        entity_states,
        active,
        safe_radius,
        braking_accel,
        velocity_margin,
    )

    # 流形残差 c = h + mu。mu 不放进环境状态，而是由算法 rollout 的
    # scan carry 单独维护，这样可以保持论文里增广状态的时间连续性。
    mu = jnp.where(active, mu, jnp.maximum(-h, slack_min))
    mu = jnp.clip(jnp.nan_to_num(mu, nan=slack_min, posinf=50.0, neginf=slack_min), slack_min, 50.0)
    c = h + mu

    # 约束漂移 psi 收集 h_dot 中不受当前 agent action 控制的部分：
    # 当前 agent 的被动动力学，以及邻居实体的运动。
    psi_own = dh_down @ own_drift
    psi_entities = jnp.einsum("mns,ns->m", dh_dentities, entity_drift)
    psi = psi_own + psi_entities

    # J_h G 是 h_dot = psi + J_h G u 中 action 对约束变化率的雅可比。
    action_jac = dh_down @ own_control_matrix
    slack_exp_arg = jnp.clip(slack_beta * mu, a_max=20.0)
    slack_actuation = jnp.expm1(slack_exp_arg)

    # 对 slack 坐标做缩放。slack_weight 越大，投影越倾向于先修正真实 action，
    # 而不是依赖虚拟 slack 控制来满足等式。
    j_slack = jnp.diag(slack_actuation / slack_weight)
    j_aug = jnp.concatenate([action_jac, j_slack], axis=1)
    u_ref_aug = jnp.concatenate([u_ref, jnp.zeros_like(h)])

    # 将 nominal augmented input 投影到仿射切空间条件上：
    #   psi + J_aug u_aug + contraction_gain * c = 0
    # 这里使用“最小改变量”的闭式最小二乘投影；reg 是给秩亏情况用的
    # 小 Tikhonov 正则项，避免数值不稳定。
    residual = psi + j_aug @ u_ref_aug + contraction_gain * c
    gram = j_aug @ j_aug.T + reg * jnp.eye(j_aug.shape[0], dtype=j_aug.dtype)
    correction = j_aug.T @ jnp.linalg.solve(gram, residual)
    u_safe_aug = u_ref_aug - correction
    u_safe_aug = jnp.nan_to_num(u_safe_aug, nan=0.0, posinf=1.0, neginf=-1.0)
    v_mu = u_safe_aug[action_dim:]
    # 虚拟 slack 控制只在 filter 内部积分，不写回环境。
    mu_next = mu + dt * slack_actuation * v_mu
    mu_next = jnp.clip(jnp.nan_to_num(mu_next, nan=slack_min, posinf=50.0, neginf=slack_min), slack_min, 50.0)
    mu_reset = jnp.maximum(-h, slack_min)
    mu_next = jnp.where(active, mu_next, mu_reset)
    # 只执行真实物理 action；虚拟 slack 控制最后丢弃。
    return u_safe_aug[:action_dim], mu_next


def _constraint_counts(env: LidarEnv, top_k_obs: int) -> Tuple[int, int, int]:
    n_agent_constraints = env.num_agents - 1
    n_obs_constraints = min(top_k_obs, env.params["top_k_rays"]) if env.params["n_obs"] > 0 else 0
    return n_agent_constraints, n_obs_constraints, n_agent_constraints + n_obs_constraints


def _agent_constraint_data(
        env: LidarEnv,
        graph: GraphsTuple,
        agent_states: AgentState,
        obs_states_all: Array,
        i_agent: int,
        top_k_obs: int,
        n_agent_constraints: int,
        n_obs_constraints: int,
        safety_margin: float,
) -> Tuple[AgentState, AgentState, Array, Array]:
    other_ids = [idx for idx in range(env.num_agents) if idx != i_agent]
    entities = []
    entity_drifts = []
    active = []
    safe_radius = []
    if n_agent_constraints > 0:
        # agent-agent 约束使用双倍半径，因为两个 agent 都有物理尺寸。
        other_states = agent_states[jnp.array(other_ids)]
        entities.append(other_states)
        entity_drifts.append(_entity_drift(other_states, env.state_dim))
        active.append(jnp.ones((n_agent_constraints,), dtype=bool))
        safe_radius.append(jnp.ones((n_agent_constraints,), dtype=agent_states.dtype) *
                           (2.0 * env.params["car_radius"] + safety_margin))

    if n_obs_constraints > 0:
        # agent-obstacle 约束把 lidar hit point 当作静态实体；
        # 超出通信半径的 hit point 标记为 inactive。
        obs_states = obs_states_all[i_agent, :top_k_obs]
        obs_dist = jnp.linalg.norm(obs_states[:, :2] - agent_states[i_agent, None, :2], axis=-1)
        obs_active = obs_dist < env.params["comm_radius"]
        entities.append(obs_states)
        entity_drifts.append(jnp.zeros_like(obs_states))
        active.append(obs_active)
        safe_radius.append(jnp.ones((n_obs_constraints,), dtype=agent_states.dtype) *
                           (env.params["car_radius"] + safety_margin))

    return (
        jnp.concatenate(entities, axis=0),
        jnp.concatenate(entity_drifts, axis=0),
        jnp.concatenate(active, axis=0),
        jnp.concatenate(safe_radius, axis=0),
    )


def lidar_manifold_init_slack(
        env: LidarEnv,
        graph: GraphsTuple,
        *,
        top_k_obs: int = 3,
        safety_margin: float = 0.02,
        braking_accel: float = 1.0,
        velocity_margin: float = 0.02,
        slack_min: float = 0.1,
) -> Array:
    """Initialize the private manifold slack state for one LidarEnv graph."""
    agent_states = graph.type_states(type_idx=env.AGENT, n_type=env.num_agents)
    n_agent_constraints, n_obs_constraints, n_constraints = _constraint_counts(env, top_k_obs)
    if n_constraints == 0:
        return jnp.zeros((env.num_agents, 0), dtype=agent_states.dtype)

    obs_states_all = None
    if env.params["n_obs"] > 0:
        n_hits = env.params["top_k_rays"] * env.num_agents
        obs_states_all = graph.type_states(type_idx=env.OBS, n_type=n_hits)
        obs_states_all = obs_states_all.reshape(env.num_agents, env.params["top_k_rays"], env.state_dim)

    entity_states = []
    active_constraints = []
    safe_radius_constraints = []
    for i_agent in range(env.num_agents):
        entity_states_i, _, active_i, safe_radius_i = _agent_constraint_data(
            env, graph, agent_states, obs_states_all, i_agent, top_k_obs,
            n_agent_constraints, n_obs_constraints, safety_margin
        )
        entity_states.append(entity_states_i)
        active_constraints.append(active_i)
        safe_radius_constraints.append(safe_radius_i)

    entity_states = jnp.stack(entity_states, axis=0)
    active_constraints = jnp.stack(active_constraints, axis=0)
    safe_radius_constraints = jnp.stack(safe_radius_constraints, axis=0)
    h = jax.vmap(_pairwise_collision_h, in_axes=(0, 0, 0, 0, None, None))(
        agent_states,
        entity_states,
        active_constraints,
        safe_radius_constraints,
        braking_accel,
        velocity_margin,
    )
    return jnp.maximum(-h, slack_min)


def lidar_manifold_project(
        env: LidarEnv,
        graph: GraphsTuple,
        action_ref: Action,
        *,
        top_k_obs: int = 3,
        safety_margin: float = 0.02,
        braking_accel: float = 1.0,
        velocity_margin: float = 0.02,
        contraction_gain: float = 30.0,
        slack_min: float = 0.1,
        slack_beta: float = 1.0,
        slack_weight: float = 10.0,
        reg: float = 1e-5,
) -> Action:
    """Project LidarEnv nominal actions through a one-step manifold filter.

    This is a single-step safety filter. It treats the policy action as the
    nominal reference action and projects it using local pairwise constraints
    against other agents and per-agent lidar hit points.
    """
    slack_state = lidar_manifold_init_slack(
        env,
        graph,
        top_k_obs=top_k_obs,
        safety_margin=safety_margin,
        braking_accel=braking_accel,
        velocity_margin=velocity_margin,
        slack_min=slack_min,
    )
    action_safe, _ = lidar_manifold_project_with_slack(
        env,
        graph,
        action_ref,
        slack_state,
        top_k_obs=top_k_obs,
        safety_margin=safety_margin,
        braking_accel=braking_accel,
        velocity_margin=velocity_margin,
        contraction_gain=contraction_gain,
        slack_min=slack_min,
        slack_beta=slack_beta,
        slack_weight=slack_weight,
        reg=reg,
    )
    return action_safe


def lidar_manifold_project_with_slack(
        env: LidarEnv,
        graph: GraphsTuple,
        action_ref: Action,
        slack_state: Array,
        *,
        top_k_obs: int = 3,
        safety_margin: float = 0.02,
        braking_accel: float = 1.0,
        velocity_margin: float = 0.02,
        contraction_gain: float = 30.0,
        slack_min: float = 0.1,
        slack_beta: float = 1.0,
        slack_weight: float = 10.0,
        reg: float = 1e-5,
) -> Tuple[Action, Array]:
    """Project actions and roll the private manifold slack state forward."""
    agent_states = graph.type_states(type_idx=env.AGENT, n_type=env.num_agents)
    dynamics = env.agent_control_affine_dynamics(agent_states)
    action_ref = env.clip_action(action_ref)

    # 每个 agent 约束所有其他 agent，以及自己的 top-k lidar 返回点。
    # lidar 返回点是障碍物边界 hit point，因此这里把非圆形障碍物近似为
    # 局部点障碍集合，而不是精确的矩形 SDF 约束。
    n_agent_constraints, n_obs_constraints, n_constraints = _constraint_counts(env, top_k_obs)

    if n_constraints == 0:
        return action_ref, slack_state

    obs_states_all = None
    if env.params["n_obs"] > 0:
        # LidarEnv 把障碍物观测存成每个 agent 的 hit-point nodes：
        # (n_agents * top_k_rays, state_dim)。这里 reshape 回每个 agent
        # 自己的局部 lidar 返回点，再组装约束。
        n_hits = env.params["top_k_rays"] * env.num_agents
        obs_states_all = graph.type_states(type_idx=env.OBS, n_type=n_hits)
        obs_states_all = obs_states_all.reshape(env.num_agents, env.params["top_k_rays"], env.state_dim)

    entity_states = []
    entity_drifts = []
    active_constraints = []
    safe_radius_constraints = []

    for i_agent in range(env.num_agents):
        entity_states_i, drift_entities_i, active_i, safe_radius_i = _agent_constraint_data(
            env, graph, agent_states, obs_states_all, i_agent, top_k_obs,
            n_agent_constraints, n_obs_constraints, safety_margin
        )
        entity_states.append(entity_states_i)
        entity_drifts.append(drift_entities_i)
        active_constraints.append(active_i)
        safe_radius_constraints.append(safe_radius_i)

    entity_states = jnp.stack(entity_states, axis=0)
    entity_drifts = jnp.stack(entity_drifts, axis=0)
    active_constraints = jnp.stack(active_constraints, axis=0)
    safe_radius_constraints = jnp.stack(safe_radius_constraints, axis=0)

    action_safe, slack_next = jax.vmap(
        _single_agent_project,
        in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    )(
        env.action_dim,
        env.dt,
        agent_states,
        dynamics.drift,
        dynamics.control_matrix,
        entity_states,
        entity_drifts,
        active_constraints,
        safe_radius_constraints,
        slack_state,
        action_ref,
        braking_accel,
        velocity_margin,
        contraction_gain,
        slack_min,
        slack_beta,
        slack_weight,
        reg,
    )
    assert action_safe.shape == action_ref.shape
    assert slack_next.shape == slack_state.shape
    return jnp.nan_to_num(env.clip_action(action_safe), nan=0.0, posinf=1.0, neginf=-1.0), slack_next
