from dataclasses import dataclass
from typing import Literal


GeometryMode = Literal["oriented_box", "ellipse"]
ModelProfile = Literal["lightweight", "dgppo_default"]
BackgroundBehavior = Literal["expert", "static"]


@dataclass(frozen=True)
class GPUDriveDGPPOConfig:
    """Algorithm-facing configuration fixed by the GPUDrive design spec."""

    num_controlled: int = 5
    num_background: int = 10
    control_all_vehicles: bool = False
    use_native_controlled_mask: bool = False
    keep_all_scene_vehicles: bool = False
    background_behavior: BackgroundBehavior = "expert"
    max_scene_vehicles: int = 64
    num_road_segments: int = 32
    road_feature_dim: int = 8
    critic_max_neighbors: int = 10
    num_lidar_samples: int = 50
    rollout_horizon: int = 64
    safety_margin_m: float = 0.5
    initial_lane_max_distance_m: float = 4.0
    intersection_min_start_radius_m: float = 8.0
    intersection_max_start_radius_m: float = 65.0
    intersection_min_goal_radius_m: float = 12.0
    intersection_max_goal_radius_m: float = 90.0
    intersection_min_trip_m: float = 15.0
    intersection_max_trip_m: float = 110.0
    intersection_min_arm_angle_rad: float = 0.75
    intersection_min_route_angle_rad: float = 0.70
    straight_max_heading_difference_rad: float = 0.35
    straight_max_cluster_length_m: float = 45.0
    straight_max_cluster_width_m: float = 18.0
    straight_min_vehicle_spacing_m: float = 8.0
    straight_max_background_radius_m: float = 55.0
    straight_max_background_lateral_m: float = 18.0
    straight_background_endpoint_margin_m: float = 3.0
    straight_min_goal_distance_m: float = 20.0
    straight_max_goal_distance_m: float = 85.0
    goal_progress_scale_m: float = 10.0
    goal_progress_clip: float = 0.1
    goal_progress_weight: float = 1.0
    lane_deviation_weight: float = 0.01
    lane_deviation_clip_m: float = 4.0
    goal_bonus: float = 1.0
    vehicle_collision_weight: float = -0.75
    offroad_weight: float = -0.75
    geometry: GeometryMode = "oriented_box"
    model_profile: ModelProfile = "lightweight"
    use_rnn: bool = False
    observation_radius_m: float = 50.0
    max_speed_mps: float = 40.0
    max_vehicle_length_m: float = 12.0
    max_vehicle_width_m: float = 4.0
    max_vehicle_height_m: float = 5.0

    def __post_init__(self):
        if (not self.control_all_vehicles
                and (self.num_controlled != 5 or self.num_background != 10)):
            raise ValueError("the GPUDrive baseline requires exactly 5 controlled + 10 observed neighbors")
        if self.control_all_vehicles and self.num_controlled != self.max_scene_vehicles:
            raise ValueError("all-vehicle control requires one controlled slot per scene slot")
        if self.num_lidar_samples != 50:
            raise ValueError("the initial baseline keeps GPUDrive's compiled 50-ray LiDAR configuration")
        if self.rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")
        if self.safety_margin_m < 0:
            raise ValueError("safety_margin_m must be non-negative")
        if self.initial_lane_max_distance_m <= 0 or self.goal_progress_scale_m <= 0:
            raise ValueError("lane threshold and goal progress scale must be positive")
        if self.goal_progress_weight < 0:
            raise ValueError("goal progress weight must be non-negative")
        if not (0 < self.intersection_min_start_radius_m < self.intersection_max_start_radius_m):
            raise ValueError("invalid intersection start-radius interval")
        if not (0 < self.intersection_min_goal_radius_m < self.intersection_max_goal_radius_m):
            raise ValueError("invalid intersection goal-radius interval")
        if not (0 < self.intersection_min_trip_m < self.intersection_max_trip_m):
            raise ValueError("invalid intersection trip-length interval")
        if not (0 < self.straight_min_goal_distance_m < self.straight_max_goal_distance_m):
            raise ValueError("invalid straight goal-distance interval")
        if self.straight_min_vehicle_spacing_m <= 0:
            raise ValueError("straight vehicle spacing must be positive")
        if self.straight_max_background_lateral_m <= 0:
            raise ValueError("straight background lateral distance must be positive")
        if self.straight_background_endpoint_margin_m < 0:
            raise ValueError("straight background endpoint margin must be non-negative")

    @property
    def num_vehicles(self) -> int:
        return self.num_controlled + self.num_background

    @property
    def n_cost(self) -> int:
        return 2  # vehicle collision, RoadEdge collision

    @property
    def max_neighbors(self) -> int:
        return 10 if self.control_all_vehicles else self.num_background

    @property
    def gnn_layers(self) -> tuple[int, int, int]:
        """Return actor, task-critic and safety-critic depths."""
        if self.model_profile == "lightweight":
            return 1, 1, 1
        if self.model_profile == "dgppo_default":
            return 2, 2, 1
        raise ValueError(f"unknown model profile: {self.model_profile}")
