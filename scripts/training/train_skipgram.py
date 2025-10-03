import math
import random
import time
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import torch
from humetrix.skipgram import SGNS, build_vocabulary
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


class SkipGramDataset(IterableDataset):
    def __init__(self, filepath, chunk_size=1000000, number_examples=None):
        super().__init__()
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.seed = int(time.time())

        self.lazy_frame = pl.scan_parquet(filepath)
        if number_examples is not None:
            self.lazy_frame = self.lazy_frame.head(number_examples)
        self.length = self.lazy_frame.select(pl.len()).collect().item()
        self._chunk_ids = self._shuffle_chunks()

    def __iter__(self):
        for chunk_id in self._chunk_ids:
            chunk_df = (self.lazy_frame
                        .slice(chunk_id * self.chunk_size, self.chunk_size)
                        .collect())
            chunk_df = chunk_df.sample(fraction=1.0, shuffle=True, with_replacement=False)
            yield from chunk_df.iter_rows()

    def _shuffle_chunks(self):
        num_chunks = math.ceil(self.length / self.chunk_size)
        chunk_ids = list(range(num_chunks))
        g = random.Random(self.seed)
        g.shuffle(chunk_ids)
        return chunk_ids


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--context-windows', '-c',
                        help='Corpus file (.parquet) converted to token ids and context windows. Check scripts/preprocessing/skipgram/ scripts.',
                        type=Path, required=True)
    parser.add_argument('--vocab', '-v',
                        help='Vocabulary file (.csv) with token frequency counts.',
                        type=Path, required=True)
    parser.add_argument('--output', '-o',
                        help='File to save trained model to.',
                        type=Path, required=True)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs to train',
                        type=int, required=False, default=10)
    parser.add_argument('--batch-size', '-b',
                        help='Training batch size.',
                        type=int, required=False, default=8192)
    parser.add_argument('--chunk-size', '-s',
                        help='Corpus loading chunk size.',
                        type=int, required=False, default=1000000)
    parser.add_argument('--number-examples', '-n',
                        help='Fixed number of examples to train the model on',
                        type=int, required=False, default=None)
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

    dataset = SkipGramDataset(args.context_windows, chunk_size=args.chunk_size,
                              number_examples=args.number_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_steps = epochs * (dataset.length // batch_size)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=total_steps)

    model.train()
    pbar = tqdm(total=total_steps)
    for epoch in range(epochs):
        epoch_loss = 0.0
        avg_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (targets, contexts) in enumerate(dataloader):
            targets = targets.to(device)
            contexts = torch.stack(contexts, dim=1).to(device)

            optimizer.zero_grad()

            loss = model(targets, contexts)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            avg_loss = epoch_loss / (batch_idx + 1)

            pbar.update()
            pbar.set_description_str(f'Epoch {epoch+1}/{epochs}')
            pbar.set_postfix({'Loss': f'{batch_loss:.4f}',
                              'Avg': f'{avg_loss:.4f}',
                              'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s')
    pbar.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
