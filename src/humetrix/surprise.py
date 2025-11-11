"""
This module implements the surprise-based humor metric from He et al. (2019)_[1].

References
----------
[1] He He, Nanyun Peng, and Percy Liang. 2019. Pun Generation with Surprise. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019. Association for Computational Linguistics, Minneapolis, 1734–1744. https://doi.org/10.18653/v1/N19-1172
"""

import re

import numpy as np
import spacy
from spacy.language import Language
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class LocalGlobalSurprise:
    """
    Calculate the surprise-based humor metric from He et al. (2019).

    Parameters
    ----------
    sentence : str
        The sentence to be analyzed.
    pun_sign : str
        The pun sign. The word or phrase that is ambiguous or humorous.
    alt_sign : str
        The alternative sign. The word or phrase that is evoked by the pun sign.
    """

    def __init__(
        self, sentence: str, pun_sign: str, alt_sign: str, spacy_model: Language
    ):
        super().__init__()
        self.text = sentence
        self.pun_sign = pun_sign
        self.alt_sign = alt_sign
        self.spacy_model = spacy_model

    def surprisal(
        self, context: str, tokenizer: PreTrainedTokenizer, lm: PreTrainedModel
    ) -> float:
        """
        Calculate the surprisal of the pun sign given a context.

        Parameters
        ----------
        context : str
            The context to use for calculating surprisal.
        tokenizer : transformers.PreTrainedTokenizer
            The transformer tokenizer to use.
        lm : transformers.PreTrainedModel
            The language model to retrieve probabilities from.

        Returns
        -------
        float
            The surprisal score.
        """
        sign_regex_variations = [
            rf'\b{re.escape(self.pun_sign)}\b',
            rf'\b{re.escape(self.pun_sign)}',
            rf'{re.escape(self.pun_sign)}',
        ]
        for sign_regex in sign_regex_variations:
            sign_loc = re.search(sign_regex, context, re.IGNORECASE)
            if sign_loc:
                break
        sign_start, sign_end = sign_loc.span()
        masked_txt = ''.join(
            (
                context[:sign_start],
                f'{tokenizer.mask_token}',
                context[sign_end:],
            )
        )

        lm.eval()
        torch.set_grad_enabled(False)

        # Prepare input
        first_layer_name = list(lm.hf_device_map.keys())[0]
        device = lm.hf_device_map[first_layer_name]

        token_ids = tokenizer.encode(masked_txt, return_tensors='pt').to(device)
        masked_position = token_ids.squeeze() == tokenizer.mask_token_id
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
        return -np.log(pun_sign_prob / alt_sign_prob)

    def get_local_context(self, window_size: int) -> str:
        """
        Get the local context around the pun sign.

        Parameters
        ----------
        window_size : int
            The size of the window around the pun sign.

        Returns
        -------
        str
            The local context.
        """
        sign_regex_variations = [
            rf'\b{re.escape(self.pun_sign)}\b',
            rf'\b{re.escape(self.pun_sign)}',
            rf'{re.escape(self.pun_sign)}',
        ]
        for sign_regex in sign_regex_variations:
            sign_loc = re.search(sign_regex, self.text, re.IGNORECASE)
            if sign_loc:
                break
        sign_start, sign_end = sign_loc.span()

        doc = self.spacy_model(self.text)
        tokens_idx = [
            i
            for i, token in enumerate(doc)
            if token.idx < sign_end and (token.idx + len(token)) > sign_start
        ]
        tokens_start = max(0, tokens_idx[0] - window_size)
        tokens_end = min(len(doc), tokens_idx[-1] + window_size)
        return ''.join(
            [
                token.text_with_ws
                for i, token in enumerate(doc)
                if i >= tokens_start and i <= tokens_end
            ]
        )

    def score(
        self, tokenizer: PreTrainedTokenizer, lm: PreTrainedModel
    ) -> float:
        """
        Calculate the local-global surprise score.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            The transformer tokenizer to use.
        lm : transformers.PreTrainedModel
            The language model to retrieve probabilities from.

        Returns
        -------
        float
            The local-global surprise score. If any error occurs, returns -1.
        """
        if self.pun_sign.lower() not in self.text.lower():
            return -1
        local_context = self.get_local_context(2)
        global_s = self.surprisal(self.text, tokenizer, lm)
        local_s = self.surprisal(local_context, tokenizer, lm)

        if global_s <= 0 or local_s < 0:
            return -1
        return local_s / global_s
