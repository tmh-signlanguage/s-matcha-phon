import time
import logging
from lightning.pytorch.callbacks import Callback
from src_mask_fast.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


# log = logging.getLogger(__name__)

class EpochTimer(Callback):
    """Logs wall-clock time per training epoch.

    _epoch_start is None whenever we have no start stamp for the epoch that is
    ending. That happens on resume: Lightning restores the loop mid-epoch and
    calls on_train_epoch_end without on_train_epoch_start ever running in this
    process, which used to raise AttributeError. A partial epoch has no
    meaningful duration, so we skip it rather than log a bogus number.
    """

    def __init__(self):
        super().__init__()
        self._epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        # monotonic, not time(): immune to NTP steps, and we only need a delta.
        self._epoch_start = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is None:
            log.info(
                f"Epoch {trainer.current_epoch} ended without a start stamp "
                f"(resumed mid-epoch); skipping epoch-time log."
            )
            return
        elapsed = time.monotonic() - self._epoch_start
        self._epoch_start = None
        pl_module.log("time/epoch_seconds", elapsed, on_epoch=True)
        log.info(f"Epoch {trainer.current_epoch} took {elapsed:.2f}s ({elapsed/60:.2f} min)")
