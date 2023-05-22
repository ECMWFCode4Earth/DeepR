from labml import experiment

from deepr.model.configs import Configs


def main():
    experiment.create(name="diffuse", writers={"screen", "lambl"})

    configs = Configs()
    configs.init()

    experiment.configs(
        configs, {"dataset": "reanalysis-SR-tas", "image_channels": 1, "epochs": 100}
    )
    experiment.add_pytorch_models({"eps_model": configs.eps_model})

    with experiment.start():
        configs.run()


if __name__ == "__main__":
    main()
