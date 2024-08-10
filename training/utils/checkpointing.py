import copy
from pathlib import Path
import torch


class CheckpointManager:
    def __init__(
        self, output_dir: str = "/tmp", keep_recent: int = 50, **checkpointables
    ):
        """
        Args:
            output_dir: Path to a directory to save checkpoints.
            keep_recent: Number of recent `k` checkpoints to keep, old checkpoints
                will be deleted. Set to a very large value to keep all checkpoints.
            checkpointables: Keyword arguments with any checkpointable objects, for
                example: model, optimizer, LR scheduler, AMP gradient scaler.
        """
        self.output_dir = Path(output_dir)
        self.keep_recent = keep_recent
        self._recent_iterations = []

        # Shallow copy, keeps references to tensors as original objects so they
        # can be updated externally, or loaded in here without needing explicit
        # synchronization after every operation.
        self.checkpointables = copy.copy(checkpointables)

    def step(self, iteration: int, best_validation_loss: float, best_recall_sum: float, patience: int):
        """
        Save a checkpoint; keys match those in :attr:`checkpointables`.

        Args:
            iteration: Current training iteration. Will be saved with other
                checkpointables.
        """
        out_state_dict = {}
        for key in self.checkpointables:
            out_state_dict[key] = self.checkpointables[key].state_dict()

        out_state_dict['best_validation_loss'] = best_validation_loss
        out_state_dict['best_recall_sum'] = best_recall_sum
        out_state_dict['patience'] = patience
        out_state_dict['last_epoch'] = iteration

        # String formatting, assuming we won't train for more than 99M iterations.
        iter_str = f"{iteration:0>8d}"

        # Save checkpoint corresponding to current iteration.
        torch.save(out_state_dict, self.output_dir / f"checkpoint_{iter_str}.pth")
        with (self.output_dir / "last_checkpoint.txt").open("w") as f:
            f.write(f"checkpoint_{iter_str}.pth")

        # Remove earliest checkpoint if there are more on disk.
        self._recent_iterations.append(iter_str)
        if len(self._recent_iterations) > self.keep_recent:
            oldest_iteration = self._recent_iterations.pop(0)
            (self.output_dir / f"checkpoint_{oldest_iteration}.pth").unlink()
    
    def best(self, best_type: str, iteration: int, best_validation_loss: float, best_recall_sum: float, patience: int):
        """
        Save and overwrites the best checkpoint with name `checkpoint_best.pth`. This method
        does not update `last_checkpoint.txt` or delete the oldest checkpoint.
        """

        out_state_dict = {}
        for key in self.checkpointables:
            out_state_dict[key] = self.checkpointables[key].state_dict()

        out_state_dict['best_validation_loss'] = best_validation_loss
        out_state_dict['best_recall_sum'] = best_recall_sum
        out_state_dict['patience'] = patience
        out_state_dict['last_epoch'] = iteration

        # Save checkpoint corresponding to current iteration.
        torch.save(out_state_dict, self.output_dir / f"checkpoint_best_{best_type}.pth")

    def resume(self, mode: str) -> int:
        """
        Find the last saved checkpoint in :attr:`output_dir` (from a previous job)
        and load it to resume the job. This method will log a warning message if
        no checkpoint is found for loading.
        """

        print(f"Attempting to resume job from {self.output_dir}...")
        
        if mode=='last': # Load last checkpoint
            last_ckpt_info_file = self.output_dir / "last_checkpoint.txt"
            if last_ckpt_info_file.exists():
                ckpt_path = last_ckpt_info_file.read_text().strip()
                print(f"Found last checkpoint in {self.output_dir}: {ckpt_path}")
                return self.load(self.output_dir / ckpt_path)
            else:
                print(
                    f"No checkpoint found in {self.output_dir} to resume job! "
                    "Hopefully this is the beginning of a fresh job."
                )
                return 0
        elif mode=='best_val_loss':
            best_ckpt_path = self.output_dir / "checkpoint_best_validation-loss.pth"
            if best_ckpt_path.exists():
                print(f"Found best checkpoint in {self.output_dir}: {best_ckpt_path}")
                return self.load(best_ckpt_path)
        elif mode=='best_recall':
            best_ckpt_path = self.output_dir / "checkpoint_best_recall.pth"
            if best_ckpt_path.exists():
                print(f"Found best checkpoint in {self.output_dir}: {best_ckpt_path}")
                return self.load(best_ckpt_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def load(self, path: str | Path) -> int:
        """
        Load a saved checkpoint from a given file path. This method tries to find
        each of :attr:`checkpointables` in the file and load their state dict.

        Args:
            path: Path to a directory/checkpoint saved by :meth:`step`.

        Returns:
            [iteration, best_validation_loss, best_recall_sum, patience]
            Iteration is the epoch number corresponding to the loaded checkpoint (to resume training).
            If iteration is not found in file, this method will return -1 on all the variables.
        """

        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu")
        iteration = checkpoint.pop("last_epoch", -1)
        best_validation_loss = checkpoint.pop("best_validation_loss", -1)
        best_recall_sum = checkpoint.pop("best_recall_sum", -1)
        patience = checkpoint.pop("patience", -1)

        # Keep flags of all checkpointables to lo which ones were not loaded.
        is_loaded = {key: False for key in self.checkpointables}

        # Load each checkpointable from checkpoint.
        for key in checkpoint:
            if key in self.checkpointables:
                print(f"Loading {key} from {path}")

                self.checkpointables[key].load_state_dict(checkpoint[key])

                is_loaded[key] = True
            else:
                print(f"{key} not found in `checkpointables`.")

        not_loaded: list[str] = [key for key in is_loaded if not is_loaded[key]]
        if len(not_loaded) > 0:
            print(f"Checkpointables not found in file: {not_loaded}")
        return iteration, best_validation_loss, best_recall_sum, patience
