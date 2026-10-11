"""Train the JAX GNN-IPPO baseline on GPUDrive."""

import sys

from train_gpudrive import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    args.algorithm = "gnn_ippo"
    args.control_all_vehicles = True
    args.keep_all_vehicles = True
    args.native_controlled_mask = True
    # Match the checked-in SB3-IPPO example dataset unless explicitly
    # overridden.  All four source scenes participate instead of pinning 402.
    if "--scene-file" not in sys.argv:
        args.scene_file = "tfrecord"
    if "--dataset-size" not in sys.argv:
        args.dataset_size = 4
    main(args)
