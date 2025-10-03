from typing import Any, Dict, Optional, Union
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import hydra
from functools import partial
import torch

class PLDataModule(LightningDataModule):
    """
    A `LightningDataModule` for custom datasets with supervision ratios.

    This module implements the key methods required by PyTorch Lightning:
    - prepare_data
    - setup
    - train_dataloader
    - val_dataloader
    - test_dataloader
    - teardown
    - state_dict
    - load_state_dict

    It also handles supervision ratios and custom sampling for semi-supervised learning.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        **kwargs
    ) -> None:
        """Initialize the CustomLightningDataModule.

        :param dataset_name_or_path: The name or path of the dataset to load.
        :param supervision_ratio: The supervision ratio for the training set.
        :param data_type_sampling_probability: The sampling probabilities for different data types.
        :param batch_size: The batch size. Defaults to 32.
        :param num_workers: The number of workers for data loading. Defaults to 0.
        :param pin_memory: Whether to pin memory in data loaders. Defaults to False.
        :param seed: Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.kwargs = kwargs

        self.data_train: Union[Dataset, Dict[str, Any]] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and set up datasets. This method is called on every GPU in distributed training.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = hydra.utils.instantiate(self.hparams.dataset.class_path, _convert_="all")
            
            self.data_train = {key: dataset[key] for key in dataset.keys() if not key in ['test', 'validation']}
            self.data_val = dataset['validation']
            self.data_test = dataset['test']

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """

        # collate_fn = partial(self.collate_fn, tokenizers=self.tokenizers)
        dataloaders = {}
        for key, value in self.data_train.items():
            dataloaders[key] = DataLoader(
            dataset=value,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=True,
            persistent_workers=True
        )
        return dataloaders
        # return self._get_dataloader(self.data_train)

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._get_dataloader(self.data_val)

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._get_dataloader(self.data_test)

    def _get_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Helper method to create a dataloader for validation and test sets.

        :param dataset: The dataset to create a dataloader for.
        :return: A DataLoader instance.
        """
        # collate_fn = partial(self.collate_fn, tokenizers=self.tokenizers)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after fit or test.

        :param stage: The stage being torn down ('fit', 'validate', 'test', or 'predict').
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    @staticmethod
    def collate_fn(batch):
        outputs = {}
        for keys in batch[0].keys():
            outputs[keys] = torch.stack([torch.tensor(item[keys]) for item in batch])
        return outputs

if __name__ == "__main__":
    _ = PLDataModule("path/to/dataset", [0.8, 0.1], [0.7, 0.3])