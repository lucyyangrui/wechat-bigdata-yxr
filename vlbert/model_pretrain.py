import sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskdata import MaskLM, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead, BertPreTrainedModel, BertEmbeddings, \
    BertEncoder
from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args, task=None, init_from_pretrain=True):
        super().__init__()
        if task is None:
            task = ['tag', 'mlm', 'itm']
        bert_config = BertConfig.from_pretrained(args.bert_dir)

        # this is the embedding size[1]   [bs, hidden_size]
        self.newfc_hidden = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        self.task = set(task)
        if 'tag' in task:
            self.newfc_tag = nn.Linear(bert_config.hidden_size, len(CATEGORY_ID_LIST))

        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            self.vocab_size = bert_config.vocab_size

        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = nn.Linear(bert_config.hidden_size, 1)

        if init_from_pretrain:
            self.bert_model = VLBertForMaskedLM.from_pretrained(args.pretrain_model, config=bert_config)
        else:
            self.bert_model = VLBertForMaskedLM(bert_config)

    def forward(self, task, text_input, text_mask, video_feature, video_mask, label=None, inference=False):
        masked_lm_loss, itm_loss = 0, 0
        loss, pred, accuracy = 0, None, 0

        if task is None:
            task = self.task
        return_mlm = False
        if 'mlm' in task:
            return_mlm = True
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input.cpu())
            text_input = input_ids.to(text_input.device)
            lm_label = lm_label[:, 1:].to(text_input.device)

        if 'itm' in task:
            input_feature, video_text_match_label, video_mask = self.sv.torch_shuf_video(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)
            video_mask = video_mask.to(video_feature.device)

        #  [bs, 1 + seq_len + frame_len, hidden_size] [bs, seq_len, vocab_size]
        features, lm_prediction_scores = self.bert_model(text_input, text_mask, video_feature, video_mask,
                                                         return_mlm=return_mlm)
        features_mean = torch.mean(features, 1)
        embedding = self.newfc_hidden(features_mean)

        # normed_embedding = F.normalize(embedding, p=2, dim=1)

        # cal loss
        if 'mlm' in task:
            if 'itm' in task:
                bs_not_change = torch.nonzero(video_text_match_label, as_tuple=False).view(-1)
                lm_prediction_scores = lm_prediction_scores[bs_not_change]
                lm_label = lm_label[bs_not_change]
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss / len(task) / 1.25

        if 'itm' in task:
            # [bs, 1, 1]
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / len(task) * 10

        if inference:
            pred = self.newfc_tag(embedding)
            return torch.argmax(pred, dim=1), pred
        elif 'tag' in task:
            pred = self.newfc_tag(embedding)
            return self.cal_loss(pred, label)
        return loss, masked_lm_loss, itm_loss


    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id


class VLBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(768, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        self.dropout = nn.Dropout(0.2)  ############## new

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # ???
    # def _prune_heads(self, heads_to_prune):
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, text_input, text_mask, video_feature, video_mask):
        # [bs, seq_len] -- [bs, seq_len, hidden_size]
        text_emb = self.embeddings(input_ids=text_input)

        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        video_feature = self.video_fc(video_feature)
        video_emb = self.video_embeddings(inputs_embeds=video_feature)

        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)
        
        embedding_output = self.dropout(embedding_output)   ################## new

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs


class VLBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = VLBert(config)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, text_input, text_mask, video_feature, video_mask, return_mlm=False):
        encoder_outputs = self.bert(text_input, text_mask, video_feature, video_mask)
        if return_mlm:
            # [bs, 1 + seq_len + video_frame_len, hidden_size]  [bs, seq_len, vocab_size]
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]:, :]
        else:
            return encoder_outputs, None



