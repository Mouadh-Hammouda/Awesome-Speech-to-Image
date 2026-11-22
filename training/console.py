"""What training shows in the terminal: one table row per epoch, and one progress bar underneath."""

import time

from tqdm import tqdm

# The column names are printed again after this many rows, so they stay on screen.
HEADER_EVERY = 10
HEADER = (
    "                   loss          recall@1       val recall    median  class\n"
    "epoch   time   train     val   train    val      @5    @10      rank   P@10\n"
    "---------------------------------------------------------------------------"
)


class TrainingDisplay:
    def __init__(self, epochs, start_epoch, steps_per_epoch):
        self.epochs = epochs
        self.epoch = start_epoch
        self.first_epoch = start_epoch
        self.rows = 0
        self.started = time.monotonic()
        # disable=None hides the bar when the output goes to a file, which leaves just the table.
        self.bar = tqdm(total=steps_per_epoch, desc=f"epoch {start_epoch}/{epochs} train", bar_format="{desc} |{bar:20}| {percentage:3.0f}%{postfix}", leave=False, dynamic_ncols=True, disable=None)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.bar.close()

    def start_epoch(self, epoch):
        if self.rows % HEADER_EVERY == 0:
            tqdm.write(f"\n{HEADER}")
        self.epoch = epoch
        # The label changes before the reset redraws the bar, so the old one never shows at 0%.
        self.set_stage("train", refresh=False)
        self.bar.reset()

    def set_stage(self, stage, refresh=True):
        self.bar.set_description_str(f"epoch {self.epoch}/{self.epochs} {stage:<5}", refresh=refresh)

    def batch_done(self, loss):
        status = f"loss {loss:.3f}"
        finished = self.epoch - self.first_epoch
        # The time left is known once one epoch of this session has finished.
        if finished:
            seconds_per_epoch = (time.monotonic() - self.started) / (finished + self.bar.n / self.bar.total)
            seconds_left = seconds_per_epoch * (self.epochs - self.epoch + 1 - self.bar.n / self.bar.total)
            status += f", {seconds_left / 60:.0f} min left"
        self.bar.set_postfix_str(status, refresh=False)
        self.bar.update()

    def epoch_done(self, seconds, train_loss, train, val, is_best):
        tqdm.write(
            f"{self.epoch:>5}{seconds:>6.0f}s{train_loss:>8.3f}{val['loss']:>8.3f}"
            f"{train['t2i R@1']:>8.1f}{val['t2i R@1']:>7.1f}{val['t2i R@5']:>8.1f}{val['t2i R@10']:>7.1f}"
            f"{val['t2i median rank']:>10.0f}{val['class P@10']:>7.1f}" + ("  *" if is_best else "")
        )
        self.rows += 1


def print_block(title, rows):
    # A title, then aligned name and value pairs.
    print(title)
    width = max(len(name) for name in rows)
    for name, value in rows.items():
        print(f"  {name:<{width}}  {value}")
    print(flush=True)
