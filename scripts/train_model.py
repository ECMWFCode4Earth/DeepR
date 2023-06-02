from argparse import ArgumentParser
from pathlib import Path

from deepr.workflow import MainPipeline


def main(cfg_path: Path):
    main_pipeline = MainPipeline(cfg_path)
    main_pipeline.run_pipeline()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="train_model.py",
        description="Train a Super Resolution model and validate its results",
    )
    parser.add_argument("--cfg_path", default="../resources/configuration.yml", type=Path, help="Path to the configuration file.")

    args = parser.parse_args()
    main(cfg_path=Path(args.cfg_path))
