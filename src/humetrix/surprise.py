import argparse
import re

import numpy as np
import spacy
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer)


class LocalGlobalSurprise():
    """Surprise scoring based on He et al. (2019)"""

    def __init__(self, sentence: str,
                 pun_sign: str,
                 alt_sign: str):
        super().__init__()
        self.text = sentence
        self.pun_sign = pun_sign
        self.alt_sign = alt_sign

    def surprisal(self, context: str, tokenizer: PreTrainedTokenizer, lm: PreTrainedModel) -> float:
        sign_regex_variations = [rf'\b{re.escape(self.pun_sign)}\b',
                                 rf'\b{re.escape(self.pun_sign)}',
                                 rf'{re.escape(self.pun_sign)}']
        for sign_regex in sign_regex_variations:
            sign_loc = re.search(sign_regex, context, re.IGNORECASE)
            if sign_loc:
                break
        sign_start, sign_end = sign_loc.span()
        masked_txt = ''.join((context[:sign_start],
                              f'{tokenizer.mask_token}',
                              context[sign_end:]))

        lm.eval()
        torch.set_grad_enabled(False)

        # Prepare input
        token_ids = tokenizer.encode(masked_txt, return_tensors='pt')
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id)
        masked_position = masked_position.nonzero().item()

        # Get global probabilities
        output = lm(token_ids)
        last_hidden_state = output[0].squeeze(0)
        mask_hidden_state = last_hidden_state[masked_position]

        softmax = torch.nn.Softmax(dim=0)
        probs = softmax(mask_hidden_state)
        pun_sign_ids = tokenizer.encode(self.pun_sign)[1:-1]
        pun_sign_prob = torch.prod(probs[pun_sign_ids]).item()
        alt_sign_ids = tokenizer.encode(self.alt_sign)[1:-1]
        alt_sign_prob = torch.prod(probs[alt_sign_ids]).item()
        if alt_sign_prob <= 0:
            return -1
        return -np.log(pun_sign_prob/alt_sign_prob)

    def get_local_context(self, window_size: int) -> str:
        sign_regex_variations = [rf'\b{re.escape(self.pun_sign)}\b',
                                 rf'\b{re.escape(self.pun_sign)}',
                                 rf'{re.escape(self.pun_sign)}']
        for sign_regex in sign_regex_variations:
            sign_loc = re.search(sign_regex, self.text, re.IGNORECASE)
            if sign_loc:
                break
        sign_start, sign_end = sign_loc.span()

        nlp = spacy.load('pt_core_news_sm')
        doc = nlp(self.text)
        tokens_idx = [i for i, token in enumerate(doc)
                      if token.idx < sign_end and
                      (token.idx + len(token)) > sign_start]
        tokens_start = max(0, tokens_idx[0] - window_size)
        tokens_end = min(len(doc), tokens_idx[-1] + window_size)
        return ''.join([token.text_with_ws
                        for i, token in enumerate(doc)
                        if i >= tokens_start and i <= tokens_end])

    def score(self, tokenizer, lm) -> float:
        if self.pun_sign.lower() not in self.text.lower():
            return -1
        local_context = self.get_local_context(2)
        global_s = self.surprisal(self.text, tokenizer, lm)
        local_s = self.surprisal(local_context, tokenizer, lm)

        if global_s <= 0 or local_s < 0:
            return -1
        return local_s/global_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Run script directly to test"""
    sentences = ['O que diz um coelho quando abre uma porta? Primeiro as cenouras.',
                 'O que diz um coelho quando abre uma porta? Primeiro as senhoras.',
                 'Qual é o youtuber que mais economiza na luz? O Jovem Led.',
                 'Qual é o youtuber que mais economiza na luz? O Whindersson Nunes.',
                 'Porque é que o computador não pára de espirrar? Porque apanhou um vírus.',
                 'Porque é que o computador não pára de avariar? Porque apanhou um vírus.',
                 'Qual o livro que conta a história dos imigrantes do sertão??? Vim das secas.',
                 'Qual o livro que conta a história dos imigrantes do sertão??? O quinze',
                 'O que é que acontece quando o Frodo morre? Passam-lhe uma certidão de Hobbit.',
                 'O que é que acontece quando o Frodo morre? Passam-lhe uma certidão de óbito.']
    signs = ['cenouras', 'cenouras',
             'led', 'led',
             'vírus', 'vírus',
             'vim das secas', 'vim das secas',
             'hobbit', 'hobbit']
    alt_signs = ['senhoras', 'senhoras',
                 'nerd', 'nerd',
                 'vírus', 'vírus',
                 'vidas secas', 'vidas secas',
                 'óbito', 'óbito']

    checkpoint = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    lm = AutoModelForMaskedLM.from_pretrained(checkpoint)

    print('----------------------')
    for sentence, sign, alt_sign in zip(sentences, signs, alt_signs):
        lg_surprise = LocalGlobalSurprise(sentence, sign, alt_sign)
        print(f'{sentence}\nScore: {lg_surprise.score(tokenizer, lm):.2f}')
        print('*****************')


if __name__ == '__main__':
    args = parse_args()
    main(args)
