from argparse import ArgumentParser
from pathlib import Path
import random
import time

import polars as pl
import pyarrow.parquet as pq
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from humetrix.skipgram import SGNS, build_vocabulary


class SkipGramDataset(IterableDataset):
    def __init__(self, filepath, batch_size=8192, number_examples=None):
        super().__init__()
        self.filepath = filepath
        self.batch_size = batch_size
        self.seed = int(time.time())

        self.pf = pq.ParquetFile(filepath)
        self.num_row_groups = self.pf.num_row_groups
        self.length = self.pf.metadata.num_rows

        if number_examples is not None:
            self.length = min(self.length, number_examples)

        self._group_ids = self._shuffle_groups()

    def __iter__(self):
        for group_id in self._group_ids:
            table = self.pf.read_row_group(
                group_id, columns=['token ids', 'context']
            )
            chunk_df = pl.from_arrow(table)

            chunk_df = chunk_df.sample(
                fraction=1.0,
                shuffle=True,
                seed=self.seed,
                with_replacement=False,
            )
            self.seed += 1

            # Yield pre-batched vectorized tensors directly
            for i in range(0, len(chunk_df), self.batch_size):
                batch_df = chunk_df.slice(i, self.batch_size)

                targets = batch_df['token ids'].to_torch()
                context_len = batch_df['context'][0].len()
                contexts = (
                    batch_df['context']
                    .explode()
                    .to_torch()
                    .view(-1, context_len)
                )

                yield targets, contexts

    def _shuffle_groups(self):
        group_ids = list(range(self.num_row_groups))
        g = random.Random(self.seed)
        g.shuffle(group_ids)
        return group_ids


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--context-windows',
        '-c',
        help='Corpus file (.parquet) converted to token ids and context windows. Check scripts/preprocessing/skipgram/ scripts.',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--vocab',
        '-v',
        help='Vocabulary file (.csv) with token frequency counts.',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--output',
        '-o',
        help='File to save trained model to.',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--epochs',
        '-e',
        help='Number of epochs to train',
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        '--batch-size',
        '-b',
        help='Training batch size.',
        type=int,
        required=False,
        default=8192,
    )
    parser.add_argument(
        '--number-examples',
        '-n',
        help='Fixed number of examples to train the model on',
        type=int,
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    vocab = build_vocabulary(args.vocab)
    vocab_counts = pl.read_csv(args.vocab)
    model = SGNS(vocab, 300, vocab_counts).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters())

    dataset = SkipGramDataset(
        args.context_windows,
        batch_size=batch_size,
        number_examples=args.number_examples,
    )

    dataloader = DataLoader(dataset, batch_size=None)

    total_steps = epochs * (dataset.length // batch_size)
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=1e-8, total_iters=total_steps
    )

    model.train()
    pbar = tqdm(total=total_steps)
    for epoch in range(epochs):
        epoch_loss = 0.0
        avg_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (targets, contexts) in enumerate(dataloader):
            targets = targets.to(device)
            contexts = contexts.to(device)

            optimizer.zero_grad()

            loss = model(targets, contexts)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            avg_loss = epoch_loss / (batch_idx + 1)

            pbar.update()
            pbar.set_description_str(f'Epoch {epoch + 1}/{epochs}')
            pbar.set_postfix(
                {
                    'Loss': f'{batch_loss:.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                }
            )
        epoch_time = time.time() - epoch_start_time
        print(
            f'Epoch {epoch + 1} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s'
        )
    pbar.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
