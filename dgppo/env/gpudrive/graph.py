"""Fixed-shape vehicle graphs with road-map context embedded in ego nodes."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from ...utils.graph import GraphsTuple
from ...utils.typing import Array
from .config import GPUDriveDGPPOConfig

EGO, VEHICLE = 0, 1
BASE_NODE_DIM, EDGE_DIM = 18, 7

class EgoGraphBatch(NamedTuple):
    graph: GraphsTuple
    node_valid_mask: Array
    edge_valid_mask: Array
    ego_node_ids: Array

def _vehicle_features(pos, speed, heading, length, width, height, goal,
                      controlled, ego, valid, cfg):
    x = jnp.zeros(pos.shape[:-1] + (BASE_NODE_DIM,), dtype=pos.dtype)
    x = x.at[..., 0:2].set(pos / cfg.observation_radius_m)
    x = x.at[..., 2].set(speed / cfg.max_speed_mps)
    x = x.at[..., 3].set(jnp.sin(heading)).at[..., 4].set(jnp.cos(heading))
    x = x.at[..., 5].set(length / cfg.max_vehicle_length_m)
    x = x.at[..., 6].set(width / cfg.max_vehicle_width_m)
    x = x.at[..., 7].set(height / cfg.max_vehicle_height_m)
    x = x.at[..., 8:10].set(goal / cfg.observation_radius_m)
    x = x.at[..., 11].set(controlled.astype(x.dtype)).at[..., 12].set(ego.astype(x.dtype))
    x = x.at[..., 13].set(1.).at[..., 17].set(valid.astype(x.dtype))
    return jnp.where(valid[..., None], x, 0.)

def _road_features(road, cfg):
    valid = (road[..., 6] > 0) & (jnp.sum(jnp.abs(road[..., :6]), axis=-1) > 0)
    x = jnp.zeros(road.shape[:-1] + (cfg.road_feature_dim,), dtype=road.dtype)
    if cfg.road_feature_dim == 14:
        # GPUDrive IPPO road observation: six continuous values plus a
        # seven-way road-type one-hot.  The final internal bit masks padding.
        road_type = jnp.clip(jnp.rint(road[..., 6]), 0, 6).astype(jnp.int32)
        x = x.at[..., 0:2].set(road[..., 0:2] / cfg.observation_radius_m)
        x = x.at[..., 2].set(road[..., 2] / 100.0)
        x = x.at[..., 3].set(road[..., 3] / 100.0)
        x = x.at[..., 4].set(road[..., 4] / 100.0)
        x = x.at[..., 5].set(road[..., 5] / (2 * jnp.pi))
        x = x.at[..., 6:13].set(jax.nn.one_hot(road_type, 7, dtype=road.dtype))
        x = x.at[..., 13].set(valid.astype(x.dtype))
        return jnp.where(valid[..., None], x, 0.)
    x = x.at[..., 0:2].set(road[..., 0:2] / cfg.observation_radius_m)
    x = x.at[..., 2].set(road[..., 2] / cfg.observation_radius_m)
    x = x.at[..., 3].set(road[..., 3] / 10.)
    x = x.at[..., 4].set(jnp.sin(road[..., 5])).at[..., 5].set(jnp.cos(road[..., 5]))
    x = x.at[..., 6].set(road[..., 6] / 10.).at[..., 7].set(valid.astype(x.dtype))
    return jnp.where(valid[..., None], x, 0.)

def _edge(delta, cfg):
    d = delta / cfg.observation_radius_m; r = jnp.linalg.norm(d, axis=-1); q = jnp.maximum(r, 1e-6)
    return jnp.stack([d[..., 0], d[..., 1], r, d[..., 1]/q, d[..., 0]/q,
                      jnp.ones_like(r), jnp.zeros_like(r)], axis=-1)

def build_ego_graph(ego_state, partner_obs, partner_valid, partner_controlled,
                    road_obs, ego_valid=None, config=GPUDriveDGPPOConfig()):
    ne, nn = config.num_controlled, config.max_neighbors
    if ego_state.shape != (ne, 8) or partner_obs.shape != (ne, nn, 9):
        raise ValueError("unexpected ego/partner shape")
    if road_obs.shape != (ne, config.num_road_segments, 9):
        raise ValueError("unexpected road observation shape")
    pad, n_edges = ne + ne * nn, 2 * ne * nn
    dim = BASE_NODE_DIM + config.num_road_segments * config.road_feature_dim
    nodes = jnp.zeros((pad + 1, dim), dtype=ego_state.dtype)
    types = -jnp.ones(pad + 1, dtype=jnp.int32); node_valid = jnp.zeros(pad + 1, dtype=bool)
    ev = jnp.ones(ne, dtype=bool) if ego_valid is None else ego_valid
    eb = _vehicle_features(jnp.zeros((ne,2)), ego_state[:,0], jnp.zeros(ne), ego_state[:,1],
        ego_state[:,2], ego_state[:,3], ego_state[:,4:6], ev, ev, ev, config)
    nodes = nodes.at[:ne, :BASE_NODE_DIM].set(eb)
    nodes = nodes.at[:ne, BASE_NODE_DIM:].set(_road_features(road_obs, config).reshape((ne,-1)))
    types = types.at[:ne].set(EGO); node_valid = node_valid.at[:ne].set(True)
    sends=[]; recvs=[]; edges=[]; edge_valid=[]
    for i in range(ne):
        ids = jnp.arange(ne+i*nn, ne+(i+1)*nn); p=partner_obs[i]; valid=partner_valid[i]
        pb = _vehicle_features(p[:,1:3],p[:,0],p[:,3],p[:,4],p[:,5],p[:,6],jnp.zeros((nn,2)),
            partner_controlled[i],jnp.zeros(nn,dtype=bool),valid,config)
        nodes = nodes.at[ids,:BASE_NODE_DIM].set(pb); types=types.at[ids].set(jnp.where(valid,VEHICLE,-1)); node_valid=node_valid.at[ids].set(valid)
        for j in range(nn):
            incoming_ok=valid[j]; outgoing_ok=valid[j]&partner_controlled[i,j]; nid=ids[j]
            # Every observed vehicle sends information to ego. Only another
            # controlled/cooperative vehicle receives a return message.
            sends += [jnp.where(incoming_ok,nid,pad),jnp.where(outgoing_ok,i,pad)]
            recvs += [jnp.where(incoming_ok,i,pad),jnp.where(outgoing_ok,nid,pad)]
            edges += [_edge(-p[j,1:3],config),_edge(p[j,1:3],config)]
            edge_valid += [incoming_ok,outgoing_ok]
    graph=GraphsTuple(jnp.asarray(pad+1),jnp.asarray(n_edges),jnp.clip(nodes,-1,1),jnp.clip(jnp.stack(edges),-1,1),nodes,
        jnp.stack(recvs).astype(jnp.int32),jnp.stack(sends).astype(jnp.int32),types,None)
    return EgoGraphBatch(graph,node_valid,jnp.stack(edge_valid),jnp.arange(ne))


def build_actor_scene_graph(vehicle_state, vehicle_valid, controlled_mask,
                            road_obs, vehicle_speed,
                            config=GPUDriveDGPPOConfig()):
    """MPE-style actor graph with unique vehicle nodes and masked local edges."""
    nv, k = config.max_scene_vehicles, config.max_neighbors
    dim = BASE_NODE_DIM + config.num_road_segments * config.road_feature_dim
    pad = nv
    position = vehicle_state[:, :2]
    goal = vehicle_state[:, 8:10] - position
    base = _vehicle_features(
        position, vehicle_speed, vehicle_state[:, 7], vehicle_state[:, 10],
        vehicle_state[:, 11], vehicle_state[:, 12], goal, controlled_mask,
        controlled_mask, vehicle_valid, config,
    )
    nodes = jnp.zeros((nv + 1, dim), dtype=vehicle_state.dtype)
    nodes = nodes.at[:nv, :BASE_NODE_DIM].set(base)
    road_features = _road_features(road_obs, config).reshape((nv, -1))
    nodes = nodes.at[:nv, BASE_NODE_DIM:].set(
        jnp.where(controlled_mask[:, None], road_features, 0.0)
    )
    node_type = -jnp.ones(nv + 1, dtype=jnp.int32)
    node_type = node_type.at[:nv].set(jnp.where(
        vehicle_valid, jnp.where(controlled_mask, EGO, VEHICLE), -1
    ))

    distance = jnp.linalg.norm(position[:, None] - position[None, :], axis=-1)
    candidate = (
        controlled_mask[:, None] & vehicle_valid[:, None]
        & vehicle_valid[None, :] & (~jnp.eye(nv, dtype=bool))
        & (distance <= config.observation_radius_m)
    )
    ranked = jnp.where(candidate, distance, jnp.inf)
    nearest = jnp.argsort(ranked, axis=1)[:, :k]
    senders, receivers, edges, edge_valid = [], [], [], []
    for ego in range(nv):
        for neighbor_offset in range(k):
            neighbor = nearest[ego, neighbor_offset]
            incoming = jnp.isfinite(ranked[ego, neighbor])
            outgoing = incoming & controlled_mask[neighbor]
            delta = position[ego] - position[neighbor]
            senders.extend([jnp.where(incoming, neighbor, pad),
                            jnp.where(outgoing, ego, pad)])
            receivers.extend([jnp.where(incoming, ego, pad),
                              jnp.where(outgoing, neighbor, pad)])
            edges.extend([_edge(delta, config), _edge(-delta, config)])
            edge_valid.extend([incoming, outgoing])
    graph = GraphsTuple(
        jnp.asarray(nv + 1), jnp.asarray(2 * nv * k), jnp.clip(nodes, -1, 1),
        jnp.clip(jnp.stack(edges), -1, 1), nodes,
        jnp.asarray(receivers, jnp.int32), jnp.asarray(senders, jnp.int32),
        node_type, None,
    )
    return EgoGraphBatch(
        graph, jnp.concatenate([vehicle_valid, jnp.zeros(1, dtype=bool)]),
        jnp.asarray(edge_valid), jnp.arange(nv),
    )


def build_ippo_scene_graph(vehicle_state, vehicle_valid, road_obs, vehicle_speed,
                           config=GPUDriveDGPPOConfig()):
    """Build one fixed-slot decentralized graph for GPUDrive GNN-IPPO.

    Every real vehicle keeps its scene slot for the full episode.  Candidate
    edges cover every ordered vehicle pair; pairs outside the observation
    radius are redirected to the padding node.  This deliberately avoids the
    top-k rank swapping used by the earlier GPUDrive adaptation.
    """
    nv = config.max_scene_vehicles
    if vehicle_state.shape != (nv, 14):
        raise ValueError("unexpected scene vehicle shape")
    if road_obs.shape != (nv, config.num_road_segments, 9):
        raise ValueError("unexpected road observation shape")

    dim = BASE_NODE_DIM + config.num_road_segments * config.road_feature_dim
    pad = nv
    position = vehicle_state[:, :2]
    heading = vehicle_state[:, 7]

    # GPUDrive's road observations are already ego-centric.  Express the goal
    # in that same local frame and omit absolute map position from the node.
    goal_world = vehicle_state[:, 8:10] - position
    c, s = jnp.cos(heading), jnp.sin(heading)
    goal_local = jnp.stack(
        [c * goal_world[:, 0] + s * goal_world[:, 1],
         -s * goal_world[:, 0] + c * goal_world[:, 1]], axis=-1)
    base = _vehicle_features(
        jnp.zeros_like(position), vehicle_speed, jnp.zeros_like(heading),
        vehicle_state[:, 10], vehicle_state[:, 11], vehicle_state[:, 12],
        goal_local, vehicle_valid, vehicle_valid, vehicle_valid, config,
    )
    nodes = jnp.zeros((nv + 1, dim), dtype=vehicle_state.dtype)
    nodes = nodes.at[:nv, :BASE_NODE_DIM].set(base)
    nodes = nodes.at[:nv, BASE_NODE_DIM:].set(
        _road_features(road_obs, config).reshape((nv, -1)))
    node_type = -jnp.ones(nv + 1, dtype=jnp.int32)
    node_type = node_type.at[:nv].set(jnp.where(vehicle_valid, EGO, -1))

    # Enumerate every ordered non-self pair without Python-unrolling 4032
    # copies of the same graph.  Besides compiling much faster, this makes the
    # validity mask below a single vectorized operation.
    recv = jnp.repeat(jnp.arange(nv), nv - 1)
    send_offset = jnp.tile(jnp.arange(nv - 1), nv)
    send = send_offset + (send_offset >= recv)
    delta_world = position[send] - position[recv]
    delta_local = jnp.stack([
        c[recv] * delta_world[:, 0] + s[recv] * delta_world[:, 1],
        -s[recv] * delta_world[:, 0] + c[recv] * delta_world[:, 1],
    ], axis=-1)
    distance = jnp.linalg.norm(delta_world, axis=-1)
    valid = (vehicle_valid[recv] & vehicle_valid[send]
             & (distance <= config.observation_radius_m))
    heading_delta = heading[send] - heading[recv]
    edges = jnp.stack([
        delta_local[:, 0] / config.observation_radius_m,
        delta_local[:, 1] / config.observation_radius_m,
        distance / config.observation_radius_m,
        (vehicle_speed[send] - vehicle_speed[recv]) / config.max_speed_mps,
        jnp.sin(heading_delta), jnp.cos(heading_delta),
        jnp.ones_like(distance),
    ], axis=-1)
    # GPUDrive leaves NaNs in unused vehicle slots. Redirecting an invalid
    # edge to padding is insufficient unless its attributes are masked too.
    edges = jnp.where(valid[:, None], edges, jnp.zeros_like(edges))
    senders = jnp.where(valid, send, pad).astype(jnp.int32)
    receivers = jnp.where(valid, recv, pad).astype(jnp.int32)

    n_edges = nv * (nv - 1)
    return EgoGraphBatch(
        GraphsTuple(
            jnp.asarray(nv + 1), jnp.asarray(n_edges), jnp.clip(nodes, -1, 1),
            jnp.clip(edges, -1, 1), nodes, receivers, senders,
            node_type, None,
        ),
        jnp.concatenate([vehicle_valid, jnp.zeros(1, dtype=bool)]),
        valid,
        jnp.arange(nv),
    )

def build_scene_graph(vehicle_state, vehicle_valid, controlled_mask, road_obs,
                      vehicle_speed, config=GPUDriveDGPPOConfig()):
    nv,k=config.max_scene_vehicles,config.critic_max_neighbors
    if vehicle_state.shape != (nv,14): raise ValueError("unexpected scene vehicle shape")
    dim=BASE_NODE_DIM+config.num_road_segments*config.road_feature_dim; pad=nv
    origin,yaw=vehicle_state[0,:2],vehicle_state[0,7]; c,s=jnp.cos(yaw),jnp.sin(yaw); rot=jnp.array([[c,s],[-s,c]])
    rel=(vehicle_state[:,:2]-origin)@rot.T; goal=(vehicle_state[:,8:10]-vehicle_state[:,:2])@rot.T
    base=_vehicle_features(rel,vehicle_speed,vehicle_state[:,7]-yaw,vehicle_state[:,10],vehicle_state[:,11],vehicle_state[:,12],goal,
        controlled_mask,controlled_mask,vehicle_valid,config)
    nodes=jnp.zeros((nv+1,dim),dtype=vehicle_state.dtype).at[:nv,:BASE_NODE_DIM].set(base)
    nodes=nodes.at[:config.num_controlled,BASE_NODE_DIM:].set(_road_features(road_obs,config).reshape((config.num_controlled,-1)))
    types = -jnp.ones(nv + 1, dtype=jnp.int32)
    types = types.at[:nv].set(
        jnp.where(vehicle_valid, jnp.where(controlled_mask, EGO, VEHICLE), -1)
    )
    dist=jnp.linalg.norm(rel[:,None]-rel[None,:],axis=-1)
    allowed=vehicle_valid[:,None]&vehicle_valid[None,:]&(~jnp.eye(nv,dtype=bool)); dist=jnp.where(allowed&(dist<=config.observation_radius_m),dist,jnp.inf)
    nearest=jnp.argsort(dist,axis=1)[:,:k]; sends=[]; recvs=[]; edges=[]
    for recv in range(nv):
        for j in range(k):
            send=nearest[recv,j]; ok=jnp.isfinite(dist[recv,send]); sends.append(jnp.where(ok,send,pad)); recvs.append(jnp.where(ok,recv,pad)); edges.append(_edge(rel[recv]-rel[send],config))
    return GraphsTuple(jnp.asarray(nv+1),jnp.asarray(nv*k),jnp.clip(nodes,-1,1),jnp.clip(jnp.stack(edges),-1,1),nodes,
        jnp.asarray(recvs,jnp.int32),jnp.asarray(sends,jnp.int32),types,None)
