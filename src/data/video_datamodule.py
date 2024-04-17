from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.video_dataset import VideoDataset, localization_VideoDataset
from src.data.components.data_utils import get_augmentation
from src.utils.utils import get_class_names
import os

class VideoDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: str,
        train_file: str,
        val_file: str,
        n_frames: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        input_size: int,
        split: int,
        limit_classes: int,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = get_augmentation(True, self.hparams)
        self.val_transforms = get_augmentation(False, self.hparams)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        if self.hparams.limit_classes != -1:
            return self.hparams.limit_classes
        return len(get_class_names(self.hparams.train_file, self.hparams.dataset))

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            if self.hparams.dataset=="thumos2014":
                self.data_val = localization_VideoDataset(
                    video_list=self.hparams.val_file,
                    dataset=self.hparams.dataset,
                    transform=self.val_transforms,
                    num_frames=self.hparams.n_frames,
                    limit_classes=self.hparams.limit_classes,
                )
            else:
                self.data_val = VideoDataset(
                video_list=self.hparams.val_file,
                dataset=self.hparams.dataset,
                transform=self.val_transforms,
                num_frames=self.hparams.n_frames,
                limit_classes=self.hparams.limit_classes,
            )

    def train_dataloader(self):
        return self.val_dataloader()

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass




if __name__ == "__main__":
    _ = VideoDataModule()
