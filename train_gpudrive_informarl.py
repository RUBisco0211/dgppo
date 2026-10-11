"""Independent InforMARL training entry point for GPUDrive."""

from train_gpudrive import main, parse_args


if __name__ == "__main__":
    arguments = parse_args()
    arguments.algorithm = "informarl"
    main(arguments)
