import yaml

from ml.common.device import get_device
from ml.common.seed import set_seed


def load_config():
    with open("ml/configs/base_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config


def main():
    print("=" * 50)
    print("SETQUERY ML ENVIRONMENT TEST")
    print("=" * 50)

    config = load_config()

    seed = config["training"]["seed"]

    set_seed(seed)

    device = get_device()

    print()
    print("Project:", config["project"]["name"])
    print("Seed:", seed)
    print("Batch size:", config["training"]["batch_size"])
    print("Epochs:", config["training"]["epochs"])
    print("Learning rate:", config["training"]["learning_rate"])
    print(
        "Image size:",
        config["image"]["height"],
        "x",
        config["image"]["width"],
    )
    print("Selected device:", device)

    print()
    print("ML environment is ready.")


if __name__ == "__main__":
    main()