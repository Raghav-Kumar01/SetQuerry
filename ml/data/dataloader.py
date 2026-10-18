from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
):
    """
    Create a PyTorch DataLoader.

    num_workers=0 is safest for
    Windows during development.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )