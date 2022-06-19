import torch
from typing import Any, Optional, Tuple
from transformers import AutoTokenizer


class MaskLM(object):
    def __init__(self, tokenizer_path='bert-base-chinese'):
        self.mlm_probability = 0.15
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None):
        '''
        :param inputs: input_ids
        :param special_tokens_mask: [SEP] [CLS]...
        :return: mask inputs & mask labels  [bs, seq_len] labels -- choose->labels else->-100
        '''
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # special tokens are not MASK
        if special_tokens_mask is None:
            special_tokens_mask = [
                # if id is special token, then id is T, else, id is F
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # bernoulli  the probability to choose 1/True
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% of the time, replace tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


class ShuffleVideo(object):
    def __init__(self):
        pass

    def torch_shuf_video(self, video_feature, video_mask):
        bs = video_feature.size()[0]
        # change video_feature order
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs // 2, bs))[::-1])
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        video_mask = video_mask[shuf_index]
        return video_feature, label, video_mask

