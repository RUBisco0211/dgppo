import argparse
import copy
import datetime
import os
import pickle
from pathlib import Path

# These must be set before importing modules that may import JAX/XLA.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import ipdb
import numpy as np
import wandb
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.trainer import Trainer
from dgppo.trainer.utils import is_connected


def _resolve_resume_checkpoint(resume_dir: str, resume_step: str):
    resume_dir = os.path.normpath(resume_dir)
    if os.path.exists(os.path.join(resume_dir, "actor.pkl")):
        return os.path.dirname(resume_dir), os.path.basename(resume_dir)
    if os.path.isdir(os.path.join(resume_dir, "models")):
        return os.path.join(resume_dir, "models"), resume_step
    return resume_dir, resume_step


def _checkpoint_iter(load_dir: str, step: str):
    candidate_dirs = []
    if os.path.isdir(os.path.join(load_dir, str(step))):
        candidate_dirs.append(os.path.join(load_dir, str(step)))
    if os.path.isdir(load_dir):
        candidate_dirs.append(load_dir)

    for iter_dir in candidate_dirs:
        for name in sorted(os.listdir(iter_dir), reverse=True):
            if name.startswith("latest_iter_") and name.endswith(".txt"):
                return int(name[len("latest_iter_"):-len(".txt")])

    ckpt_dir = os.path.join(load_dir, str(step))
    if not os.path.isdir(ckpt_dir):
        return None
    for name in sorted(os.listdir(ckpt_dir)):
        if name.startswith("iter_") and name.endswith(".ckpt"):
            return int(name[len("iter_"):-len(".ckpt")])
    return None


def _train_rl(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    if args.wandb_mode == "auto":
        args.wandb_mode = "online" if is_connected() else "offline"
    np.random.seed(args.seed)
    if args.debug:
        args.wandb_mode = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
    os.environ["WANDB_MODE"] = args.wandb_mode

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
    )

    horizon = env.max_episode_steps
    max_batch_size = args.n_env_train * horizon
    envs_per_minibatch = max(1, min(args.n_env_train, args.batch_size // horizon))
    adjusted_batch_size = envs_per_minibatch * horizon
    if adjusted_batch_size != args.batch_size:
        print(
            f"> Adjusting batch_size from {args.batch_size} to {adjusted_batch_size}. "
            f"It must be between {horizon} and {max_batch_size}, "
            f"and is rounded to a multiple of max_episode_steps={horizon}."
        )
        args.batch_size = adjusted_batch_size

    # create algorithm
    if (
        args.algo == "informarl_deep_qp"
        and args.deep_qp_checkpoint is None
        and args.resume_dir is None
    ):
        raise ValueError(
            "informarl_deep_qp requires --deep-qp-checkpoint for a new PPO run. "
            "Pretrain it with train_safety_filter.py."
        )
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        cbf_weight=args.cbf_weight,
        actor_gnn_layers=args.actor_gnn_layers,
        Vl_gnn_layers=args.Vl_gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        rnn_layers=args.rnn_layers,
        lr_actor=args.lr_actor,
        lr_Vl=args.lr_Vl,
        lr_Vh=args.lr_Vh,
        max_grad_norm=2.0,
        alpha=args.alpha,
        cbf_eps=args.cbf_eps,
        seed=args.seed,
        batch_size=args.batch_size,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        coef_ent=args.coef_ent,
        rnn_step=args.rnn_step,
        gamma=0.99,
        clip_eps=args.clip_eps,
        lagr_init=args.lagr_init,
        lr_lagr=args.lr_lagr,
        train_steps=args.steps,
        cbf_schedule=not args.no_cbf_schedule,
        cost_schedule=args.cost_schedule,
        manifold_top_k_obs=args.manifold_top_k_obs,
        manifold_safety_margin=args.manifold_safety_margin,
        manifold_braking_accel=args.manifold_braking_accel,
        manifold_velocity_margin=args.manifold_velocity_margin,
        manifold_contraction_gain=args.manifold_contraction_gain,
        manifold_slack_min=args.manifold_slack_min,
        manifold_slack_beta=args.manifold_slack_beta,
        manifold_slack_weight=args.manifold_slack_weight,
        manifold_reg=args.manifold_reg,
        deep_qp_checkpoint=args.deep_qp_checkpoint,
        deep_qp_gnn_layers=args.deep_qp_gnn_layers,
        deep_qp_gnn_out_dim=args.deep_qp_gnn_out_dim,
        deep_qp_hidden_dim=args.deep_qp_hidden_dim,
        deep_qp_hidden_layers=args.deep_qp_hidden_layers,
        deep_qp_lr=args.deep_qp_lr,
        deep_qp_lr_final=args.deep_qp_lr_final,
        deep_qp_tau=args.deep_qp_tau,
        deep_qp_lambda_init=(
            None
            if args.deep_qp_lambda_init is None
            else args.deep_qp_lambda_init / env.dt
        ),
        deep_qp_lambda_final=(
            None
            if args.deep_qp_lambda_final is None
            else args.deep_qp_lambda_final / env.dt
        ),
        deep_qp_lambda_decay_steps=args.deep_qp_lambda_decay_steps,
        deep_qp_constraint_scale=args.deep_qp_constraint_scale,
        deep_qp_allow_agent_count_transfer=(
            getattr(args, "deep_qp_allow_agent_count_transfer", False)
        ),
        hj_cbf_alpha=args.hj_cbf_alpha,
        hj_cbf_margin=args.hj_cbf_margin,
        hj_cbf_eps=args.hj_cbf_eps,
        gcbf_gnn_layers=args.gcbf_gnn_layers,
        gcbf_batch_size=args.gcbf_batch_size,
        gcbf_buffer_size=args.gcbf_buffer_size,
        gcbf_horizon=args.gcbf_horizon,
        gcbf_inner_epoch=args.gcbf_inner_epoch,
        gcbf_lr_actor=args.gcbf_lr_actor,
        gcbf_lr_cbf=args.gcbf_lr_cbf,
        gcbf_alpha=args.gcbf_alpha,
        gcbf_eps=args.gcbf_eps,
        gcbf_loss_action_coef=args.gcbf_loss_action_coef,
        gcbf_loss_unsafe_coef=args.gcbf_loss_unsafe_coef,
        gcbf_loss_safe_coef=args.gcbf_loss_safe_coef,
        gcbf_loss_h_dot_coef=args.gcbf_loss_h_dot_coef,
        gcbf_target_tau=args.gcbf_target_tau,
        gcbf_qp_relax_penalty=args.gcbf_qp_relax_penalty,
        gcbf_qp_chunk_size=args.gcbf_qp_chunk_size,
        gcbf_unsafe_fraction=args.gcbf_unsafe_fraction,
    )

    start_step = 0
    resume_training_state = None
    if args.resume_dir is not None:
        load_dir, load_step = _resolve_resume_checkpoint(args.resume_dir, args.resume_step)
        print(f"> Resuming weights from {os.path.join(load_dir, str(load_step))}")
        resume_iter = _checkpoint_iter(load_dir, load_step)
        if resume_iter is not None:
            start_step = resume_iter
            print(f"> Found checkpoint iteration: {resume_iter}. Target iteration: {args.steps}.")
            if start_step > args.steps:
                raise ValueError(f"resume iter {start_step} is greater than target --steps {args.steps}")
        else:
            print("> No checkpoint iteration file found; resuming from iteration 0.")
        algo.load(load_dir, load_step)
        checkpoint_dir = os.path.join(load_dir, str(load_step))
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pkl")
        algo_state_path = os.path.join(checkpoint_dir, "algo_training_state.pkl")
        if os.path.exists(trainer_state_path) and os.path.exists(algo_state_path):
            with open(trainer_state_path, "rb") as file:
                resume_training_state = pickle.load(file)
            print("> Restored optimizer and training RNG state for full-state continuation.")
        else:
            print(
                "> Legacy checkpoint has no complete training state; "
                "continuing as a parameter warm-start."
            )

    # Generate a 4 letter random identifier for the run.
    rng_ = np.random.default_rng()
    rand_id = "".join([chr(rng_.integers(65, 91)) for _ in range(4)])

    # set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m%d%H%M%S")
    if not args.debug:
        if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
            os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}", exist_ok=True)
    start_time = int(start_time)
    while os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}_{rand_id}"):
        start_time += 1

    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}_{rand_id}"
    run_name = "{}_seed{:03}_{}_{}".format(args.algo, args.seed, start_time, rand_id)
    if args.name is not None:
        run_name = "{}_{}_seed{:03}_{}_{}".format(run_name, args.name, args.seed, start_time, rand_id)
    if args.wandb_name is not None:
        run_name = args.wandb_name

    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
        "video_interval": args.eval_interval,
        "log_video": not args.no_video and not args.debug,
        "video_dpi": args.video_dpi,
        "start_step": start_step,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_run_id": args.wandb_run_id,
        "metrics_log_file": args.metrics_log_file,
        "resume_training_state": resume_training_state,
    }

    # create trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=0.99,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
    )

    # save config
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update(algo.config, allow_val_change=True)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)

    # start training
    trainer.train()


def _train_deepqp(args):
    """Run Graph-HJ pretraining and HJ-constrained InforMARL sequentially."""
    if args.resume_dir is not None:
        raise ValueError(
            "--algo deepqp starts a fresh two-stage run; use "
            "--algo informarl_deep_qp with --resume-dir to resume stage 2"
        )

    if args.wandb_mode == "auto":
        args.wandb_mode = "online" if is_connected() else "offline"
    if args.debug:
        args.wandb_mode = "disabled"

    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
    random_id = "".join(
        chr(value) for value in np.random.default_rng().integers(65, 91, size=4)
    )
    run_root = Path(args.log_dir) / args.env / "deepqp" / (
        f"seed{args.seed}_{timestamp}_{random_id}"
    )
    hj_dir = run_root / "deep-qp"
    rl_log_dir = run_root / "rl"
    metrics_log_file = run_root / "training_metrics.jsonl"
    run_root.mkdir(parents=True, exist_ok=False)
    hj_dir.mkdir()
    rl_log_dir.mkdir()

    wandb_run_id = args.wandb_run_id or (
        f"deepqp_{args.seed}_{timestamp}_{random_id}"
    )
    wandb_name = args.wandb_name or (
        f"deepqp_{args.env}_n{args.num_agents}_o{args.obs}_seed{args.seed}"
    )
    (run_root / "wandb_run_id").write_text(wandb_run_id + "\n", encoding="utf-8")

    lambda_init = (
        0.1 if args.deep_qp_lambda_init is None else args.deep_qp_lambda_init
    )
    lambda_final = (
        0.0001 if args.deep_qp_lambda_final is None else args.deep_qp_lambda_final
    )

    safety_args = argparse.Namespace(
        env=args.env,
        num_agents=args.num_agents,
        obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        seed=args.seed,
        steps=args.deep_qp_pretrain_steps,
        n_env=args.deep_qp_pretrain_n_env,
        rollout_steps=args.deep_qp_pretrain_rollout_steps,
        updates_per_collect=args.deep_qp_pretrain_updates_per_collect,
        warmup=args.deep_qp_pretrain_warmup,
        batch_size=args.deep_qp_pretrain_batch_size,
        replay_size=args.deep_qp_pretrain_replay_size,
        gnn_layers=args.deep_qp_gnn_layers,
        gnn_out_dim=args.deep_qp_gnn_out_dim,
        hidden_dim=args.deep_qp_hidden_dim,
        hidden_layers=args.deep_qp_hidden_layers,
        lr=args.deep_qp_lr,
        lr_final=args.deep_qp_lr_final,
        max_grad_norm=2.0,
        tau=args.deep_qp_tau,
        lambda_init=lambda_init,
        lambda_final=lambda_final,
        lambda_decay_steps=args.deep_qp_lambda_decay_steps,
        constraint_scale=args.deep_qp_constraint_scale,
        output_dir=str(hj_dir),
        resume=None,
        save_interval=args.deep_qp_pretrain_save_interval,
        log_interval=args.deep_qp_pretrain_log_interval,
        eval_interval=args.deep_qp_pretrain_eval_interval,
        eval_n_env=args.deep_qp_pretrain_eval_n_env,
        log_file=str(metrics_log_file.resolve()),
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_name=wandb_name,
        wandb_run_id=wandb_run_id,
        debug=args.debug,
    )

    from train_safety_filter import train as train_safety_critic

    print("> Deep-QP stage 1/2: pretraining Graph-HJ value")
    train_safety_critic(safety_args)

    rl_args = copy.deepcopy(args)
    rl_args.algo = "informarl_deep_qp"
    rl_args.deep_qp_checkpoint = str(hj_dir / "deep_qp_safety.pkl")
    rl_args.log_dir = str(rl_log_dir)
    rl_args.metrics_log_file = str(metrics_log_file.resolve())
    rl_args.wandb_run_id = wandb_run_id
    rl_args.wandb_name = wandb_name
    rl_args.name = "deepqp"

    print("> Deep-QP stage 2/2: training InforMARL with HJ constraints")
    _train_rl(rl_args)


def train(args):
    if args.algo == "deepqp":
        return _train_deepqp(args)
    return _train_rl(args)


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("-n", "--num-agents", type=int, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--obs", type=int, required=True)

    # custom arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--resume-step", type=str, default="latest")
    parser.add_argument("--cost-weight", type=float, default=0.)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument('--full-observation', action='store_true', default=False)
    parser.add_argument('--clip-eps', type=float, default=0.25)
    parser.add_argument('--lagr-init', type=float, default=0.5)
    parser.add_argument('--lr-lagr', type=float, default=1e-7)
    parser.add_argument("--cbf-weight", type=float, default=1.0)
    parser.add_argument("--cbf-eps", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--no-cbf-schedule", action="store_true", default=False)
    parser.add_argument("--cost-schedule", action="store_true", default=False)
    parser.add_argument("--no-rnn", action="store_true", default=False)

    # one-step constraint manifold filter arguments
    parser.add_argument("--manifold-top-k-obs", type=int, default=3)
    parser.add_argument("--manifold-safety-margin", type=float, default=0.02)
    parser.add_argument("--manifold-braking-accel", type=float, default=1.0)
    parser.add_argument("--manifold-velocity-margin", type=float, default=0.02)
    parser.add_argument("--manifold-contraction-gain", type=float, default=30.0)
    parser.add_argument("--manifold-slack-min", type=float, default=0.1)
    parser.add_argument("--manifold-slack-beta", type=float, default=1.0)
    parser.add_argument("--manifold-slack-weight", type=float, default=10.0)
    parser.add_argument("--manifold-reg", type=float, default=1e-5)

    # pretrained Graph-HJ critic and Deep-QP policy-training arguments
    parser.add_argument("--deep-qp-checkpoint", type=str, default=None)
    parser.add_argument("--deep-qp-gnn-layers", type=int, default=1)
    parser.add_argument("--deep-qp-gnn-out-dim", type=int, default=64)
    parser.add_argument("--deep-qp-hidden-dim", type=int, default=256)
    parser.add_argument("--deep-qp-hidden-layers", type=int, default=2)
    parser.add_argument("--deep-qp-lr", type=float, default=3e-4)
    parser.add_argument("--deep-qp-lr-final", type=float, default=3e-6)
    parser.add_argument("--deep-qp-tau", type=float, default=0.005)
    parser.add_argument("--deep-qp-lambda-init", type=float, default=None)
    parser.add_argument("--deep-qp-lambda-final", type=float, default=None)
    parser.add_argument("--deep-qp-lambda-decay-steps", type=int, default=1_000_000)
    parser.add_argument("--deep-qp-constraint-scale", type=float, default=0.5)
    parser.add_argument(
        "--deep-qp-allow-agent-count-transfer",
        action="store_true",
        default=False,
        help=(
            "allow a frozen Graph-HJ checkpoint trained with a different "
            "agent count; all other safety metadata remains strict"
        ),
    )
    parser.add_argument("--deep-qp-pretrain-steps", type=int, default=1_000_000)
    parser.add_argument("--deep-qp-pretrain-n-env", type=int, default=32)
    parser.add_argument("--deep-qp-pretrain-rollout-steps", type=int, default=32)
    parser.add_argument(
        "--deep-qp-pretrain-updates-per-collect", type=int, default=32
    )
    parser.add_argument("--deep-qp-pretrain-warmup", type=int, default=20_000)
    parser.add_argument("--deep-qp-pretrain-batch-size", type=int, default=256)
    parser.add_argument("--deep-qp-pretrain-replay-size", type=int, default=1_000_000)
    parser.add_argument("--deep-qp-pretrain-save-interval", type=int, default=10_000)
    parser.add_argument("--deep-qp-pretrain-log-interval", type=int, default=100)
    parser.add_argument("--deep-qp-pretrain-eval-interval", type=int, default=1_000)
    parser.add_argument("--deep-qp-pretrain-eval-n-env", type=int, default=8)
    parser.add_argument("--hj-cbf-alpha", type=float, default=1.0)
    parser.add_argument("--hj-cbf-margin", type=float, default=0.0)
    parser.add_argument("--hj-cbf-eps", type=float, default=0.0)

    # GCBF+ actor/CBF joint-training arguments
    parser.add_argument("--gcbf-gnn-layers", type=int, default=1)
    parser.add_argument("--gcbf-batch-size", type=int, default=256)
    parser.add_argument("--gcbf-buffer-size", type=int, default=65536)
    parser.add_argument("--gcbf-horizon", type=int, default=32)
    parser.add_argument("--gcbf-inner-epoch", type=int, default=8)
    parser.add_argument("--gcbf-lr-actor", type=float, default=3e-5)
    parser.add_argument("--gcbf-lr-cbf", type=float, default=3e-5)
    parser.add_argument("--gcbf-alpha", type=float, default=1.0)
    parser.add_argument("--gcbf-eps", type=float, default=0.02)
    parser.add_argument("--gcbf-loss-action-coef", type=float, default=1e-4)
    parser.add_argument("--gcbf-loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--gcbf-loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--gcbf-loss-h-dot-coef", type=float, default=0.01)
    parser.add_argument("--gcbf-target-tau", type=float, default=0.5)
    parser.add_argument("--gcbf-qp-relax-penalty", type=float, default=1e3)
    parser.add_argument("--gcbf-qp-chunk-size", type=int, default=32)
    parser.add_argument("--gcbf-unsafe-fraction", type=float, default=0.5)

    # NN arguments
    parser.add_argument("--actor-gnn-layers", type=int, default=2)
    parser.add_argument("--Vl-gnn-layers", type=int, default=2)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-Vl", type=float, default=1e-3)
    parser.add_argument("--lr-Vh", type=float, default=1e-3)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--rnn-step", type=int, default=16)

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--video-dpi", type=int, default=100)
    parser.add_argument(
        "--wandb-mode",
        choices=("auto", "online", "offline", "disabled"),
        default="auto",
    )
    parser.add_argument("--wandb-project", type=str, default="dgppo")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--metrics-log-file", type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
