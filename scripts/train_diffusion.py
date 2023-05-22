from labml import experiment

from deepr.model.configs import Configs


def main():
    experiment.create(name="diffuse", writers={"screen"})

    configs = Configs()

    experiment.configs(
        configs, {"dataset": "CelebA", "image_channels": 3, "epochs": 100}
    )

    configs.init()
    experiment.add_pytorch_models({"eps_model": configs.eps_model})

    with experiment.start():
        configs.run()


if __name__ == "__main__":
    main()
