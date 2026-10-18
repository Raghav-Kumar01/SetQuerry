from pathlib import Path
import csv


class ManifestReader:
    """
    Read an already-prepared dataset manifest.

    IMPORTANT:
    This class is READ ONLY.

    It does NOT:
    - clean data
    - create splits
    - modify paths
    - rename files
    - rewrite manifests
    """

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

    def get_columns(self):
        """
        Return CSV column names.
        """

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            return reader.fieldnames

    def iter_rows(self):
        """
        Read rows one at a time.

        This is memory efficient because
        we do NOT load the complete CSV
        into RAM.
        """

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                yield row

    def first_row(self):
        """
        Return only the first data row.
        """

        for row in self.iter_rows():
            return row

        return None

    def summarize_splits(
        self,
        valid_splits=("train", "val", "test")
    ):
        """
        Count train / val / test samples
        using a single pass through the CSV.
        """

        summary = {
            split: 0
            for split in valid_splits
        }

        summary["other"] = 0
        summary["total"] = 0

        for row in self.iter_rows():

            summary["total"] += 1

            split = row.get("split")

            if split in valid_splits:
                summary[split] += 1

            else:
                summary["other"] += 1

        return summary