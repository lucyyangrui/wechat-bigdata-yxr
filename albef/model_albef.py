from model_xbert import BertConfig, BertModel
from category_id_map import CATEGORY_ID_LIST

import torch
from torch import nn
import torch.nn.functional as F


class MultiModal(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config=None
                 ):
        super().__init__()

        # self.tokenizer = tokenizer
        self.distill = config['distill']

        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.num_hidden_layers = 18
        bert_config.fusion_layer = 6
        bert_config.encoder_width = 768
        
        self.drop = nn.Dropout(0.6)
        
        ## print(bert_config)
        
        print('loading model', text_encoder)

        self.text_encoder = BertModel.from_pretrained(pretrained_model_name_or_path=text_encoder, config=bert_config, add_pooling_layer=False)
        self.vision_linear = nn.Linear(bert_config.encoder_width, bert_config.encoder_width)
        # self.vision_embed = self.text_encoder.embeddings
        self.cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
        )

        self.share_cross_attention(self.text_encoder.encoder)

        if self.distill:
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
            self.vision_linear_m = nn.Linear(bert_config.encoder_width, bert_config.encoder_width)
            # self.vision_embed_m = self.text_encoder_m.embeddings
            self.share_cross_attention(self.text_encoder_m.encoder)

            self.cls_head_m = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
            )

            self.model_pairs = [[self.vision_linear, self.vision_linear_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.cls_head, self.cls_head_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, input_ids, text_mask, visual_embed, visual_mask, labels=None, alpha=0, train=True):
        visual_embed = self.vision_linear(visual_embed)
        visual_embed = self.drop(visual_embed)
        output = self.text_encoder(input_ids,
                                   attention_mask=text_mask,
                                   encoder_hidden_states=visual_embed,
                                   encoder_attention_mask=visual_mask,
                                   return_dict=True,
                                   )
        hidden_state = output.last_hidden_state[:, 0, :]
        prediction = self.cls_head(hidden_state)

        if train:
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    visual_embed_m = self.vision_linear_m(visual_embed)
                    visual_embed_m = self.drop(visual_embed_m)
                    output_m = self.text_encoder_m(input_ids,
                                                   attention_mask=text_mask,
                                                   encoder_hidden_states=visual_embed_m,
                                                   encoder_attention_mask=visual_mask,
                                                   return_dict=True,
                                                   )['last_hidden_state']
                    output_m = torch.mean(output_m, dim=1)
                    prediction_m = self.cls_head_m(output_m)

                pred_label_id = torch.argmax(prediction, dim=1)
                labels = labels.squeeze(dim=1)
                loss = (1 - alpha) * F.cross_entropy(prediction, labels) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1), dim=1).mean()
            else:
                labels = labels.squeeze(dim=1)
                pred_label_id = torch.argmax(prediction, dim=1)
                loss = F.cross_entropy(prediction, labels)

            accuracy = (labels == pred_label_id).float().sum() / labels.shape[0]
            return loss, accuracy, pred_label_id
        else:
            return torch.argmax(prediction, dim=1), prediction


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def share_cross_attention(self, model):

        for i in range(6):
            layer_num = 6 + i * 2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num + 1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias