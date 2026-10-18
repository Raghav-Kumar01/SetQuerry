from pathlib import Path
import csv

from torch.utils.data import Dataset

from ml.data.image_utils import load_image_as_tensor


class Model3Dataset(Dataset):
    """
    SEN1-2 optical + SAR dataset.

    Full dataset expected by manifest:
        ROIs1158_spring/s2_0/...
        ROIs1158_spring/s1_0/...

    A small flattened sample structure can also
    be used for smoke testing:
        ROIs1158_spring/s2_p*.png
        ROIs1158_spring/s1_p*.png

    IMPORTANT:
    The fallback is only for testing.
    The original manifest is NOT modified.
    """

    def __init__(
        self,
        manifest_path,
        dataset_root,
        split,
        max_samples=None,
        allow_flattened_sample=False
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split

        self.allow_flattened_sample = (
            allow_flattened_sample
        )

        self.samples = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["split"] != split:
                    continue

                self.samples.append(row)

                if (
                    max_samples is not None
                    and len(self.samples) >= max_samples
                ):
                    break

    def __len__(self):
        return len(self.samples)

    def _resolve_path(
        self,
        manifest_relative_path,
        sensor
    ):
        """
        First try the exact manifest path.

        If explicitly allowed, try the
        flattened 20-image sample format.
        """

        normal_path = (
            self.dataset_root
            / manifest_relative_path
        )

        if normal_path.exists():
            return normal_path

        if not self.allow_flattened_sample:
            return normal_path

        manifest_name = Path(
            manifest_relative_path
        ).name

        # Example:
        # ROIs1158_spring_s2_0_p1.png
        #
        # flattened sample:
        # s2_p1.png

        if sensor == "optical":

            patch_part = (
                manifest_name
                .split("_p")[-1]
            )

            flattened_name = (
                f"s2_p{patch_part}"
            )

        elif sensor == "sar":

            patch_part = (
                manifest_name
                .split("_p")[-1]
            )

            flattened_name = (
                f"s1_p{patch_part}"
            )

        else:
            raise ValueError(
                f"Unknown sensor: {sensor}"
            )

        flattened_path = (
            self.dataset_root
            / "ROIs1158_spring"
            / flattened_name
        )

        return flattened_path

    def __getitem__(self, index):

        sample = self.samples[index]

        optical_path = self._resolve_path(
            sample["optical_path"],
            sensor="optical"
        )

        sar_path = self._resolve_path(
            sample["sar_path"],
            sensor="sar"
        )

        optical = load_image_as_tensor(
            optical_path
        )

        sar = load_image_as_tensor(
            sar_path
        )

        return {
            "pair_id": sample["pair_id"],
            "optical": optical,
            "sar": sar,
            "season": sample["season"],
            "scene_id": sample["scene_id"],
        }