from pathlib import Path
import csv
from zipfile import ZipFile

from torch.utils.data import Dataset

from ml.data.image_utils import (
    load_image_as_tensor,
    load_image_bytes_as_tensor,
    load_binary_mask,
)


class Model2Dataset(Dataset):
    """
    LEVIR-CD dataset for Model 2.

    Input:
        before image
        after image

    Target:
        binary change mask
    """

    def __init__(
        self,
        manifest_path,
        dataset_root,
        split,
        max_samples=None
    ):
        self.manifest_path = Path(
            manifest_path
        )

        self.dataset_root = Path(
            dataset_root
        )

        self.archive_path = (
            self.dataset_root
            / "LEVIR-CD.zip"
        )

        self.archive_prefix = "LEVIR-CD/"

        self.split = split

        self.samples = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["split"] == split:

                    self.samples.append(
                        row
                    )

                    if (
                        max_samples is not None
                        and len(self.samples) >= max_samples
                    ):
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        relative_paths = [
            sample["before_image_path"],
            sample["after_image_path"],
            sample["mask_path"],
        ]

        if self.archive_path.is_file():
            archive_paths = [
                self.archive_prefix + relative_path
                for relative_path in relative_paths
            ]

            with ZipFile(self.archive_path, "r") as archive:
                before_image = load_image_bytes_as_tensor(archive.read(archive_paths[0]))
                after_image = load_image_bytes_as_tensor(archive.read(archive_paths[1]))
                mask = load_binary_mask(archive.read(archive_paths[2]))
        else:
            paths = [
                self.dataset_root / relative_path
                for relative_path in relative_paths
            ]

            before_image = load_image_as_tensor(paths[0])
            after_image = load_image_as_tensor(paths[1])
            mask = load_binary_mask(paths[2])

        return {
            "pair_id":
                sample["pair_id"],

            "before":
                before_image,

            "after":
                after_image,

            "mask":
                mask,
        }
