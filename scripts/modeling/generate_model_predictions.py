from argparse import ArgumentParser
from pathlib import Path

from deepr.workflow import MainPipeline


def main(cfg_path: Path):
    main_pipeline = MainPipeline(cfg_path)
    main_pipeline.generate_predictions()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="generate_model_predictions.py",
        description="Dump predictions from Super Resolution model",
    )
    parser.add_argument(
        "--cfg_path",
        default="../resources/configuration_predictions.yml",
        type=Path,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    main(cfg_path=Path(args.cfg_path))
