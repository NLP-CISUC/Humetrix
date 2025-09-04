import time
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import torch
from humetrix.skipgram import SGNS, build_vocabulary
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SkipGramDataset(Dataset):
    def __init__(self, filepath, chunk_size=10000):
        self.chunk_size = chunk_size
        self.lazy_frame = pl.scan_parquet(filepath)
        self.length = self.lazy_frame.select(pl.len()).collect().item()

        self._current_chunk = None
        self._chunk_start_idx = -1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        chunk_start = chunk_idx * self.chunk_size
        if self._chunk_start_idx != chunk_start:
            self._load_chunk(chunk_start)

        local_idx = idx - self._chunk_start_idx
        target = torch.tensor(self._current_chunk.item(local_idx, 'token ids'))
        context = self._current_chunk.item(local_idx, 'context').to_torch()
        return target, context

    def _load_chunk(self, start_idx):
        end_idx = min(start_idx + self.chunk_size, self.length)
        chunk_df = (self.lazy_frame
                    .slice(start_idx, end_idx - start_idx)
                    .collect())
        self._current_chunk = chunk_df
        self._chunk_start_idx = start_idx

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

    dataset = SkipGramDataset(args.context_windows)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_steps = epochs * len(dataloader)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=total_steps)

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
            pbar.set_description_str(f'Epoch {epoch+1}/{epochs}')
            pbar.set_postfix({'Loss': f'{batch_loss:.4f}',
                              'Avg': f'{avg_loss:.4f}',
                              'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s')
    pbar.close()

    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
