import torch.nn as nn
import torch
import numpy as np

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler


class CustomedBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)

    def custom_freeze_ids(self, non_frozen_ids):
        self.mask = torch.zeros(self.word_embeddings.weight.shape, dtype=torch.float32).cuda()
        for _id in non_frozen_ids:
            self.mask[_id, :] = 1
        self.word_embeddings.weight.register_hook(self.custom_backward_hook)

    def custom_backward_hook(self, grad):
        return grad * self.mask
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        return super().forward(
            input_ids=inputs_embeds,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )


class BaseBertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.first_task_embeddings = None
        self.embeddings = CustomedBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids: torch.Tensor, prompt_pool=None, x_key=None, prompt_pools=None):
        if self.first_task_embeddings is not None and x_key is None:
            embeddings_output = self.first_task_embeddings(input_ids=input_ids)
        else:
            embeddings_output = self.embeddings(input_ids=input_ids)
        return self.encoder(embeddings_output)[0]


class BertRelationEncoder(nn.Module):
    def __init__(self, config):
        super(BertRelationEncoder, self).__init__()
        self.encoder = BaseBertEncoder.from_pretrained(config.bert_path)
        if config.pattern in ["entity_marker"]:
            self.pattern = config.pattern
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.encoder.embeddings.custom_freeze_ids(list(range(config.vocab_size, config.vocab_size + config.marker_size)))
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False
        else:
            raise Exception("Wrong encoding.")
        
    def freeze_embeddings(self):
        for param_ in self.encoder.embeddings.parameters():
            param_.requires_grad = False
        for param_ in self.encoder.first_task_embeddings.parameters():
            param_.requires_grad = False
        for param_ in self.encoder.embeddings.parameters():
            param_.requires_grad = False

    def forward(self, input_ids: torch.Tensor, prompt_pool=None, x_key=None, prompt_pools=None):
        e11 = (input_ids == 30522).nonzero()
        e21 = (input_ids == 30524).nonzero()

        attention_out = self.encoder(input_ids, prompt_pool, x_key, prompt_pools)
        output = []
        for i in range(e11.shape[0]):
            if prompt_pool is not None:
                additional_length = prompt_pool.total_prompt_length
            elif prompt_pools is not None:
                additional_length = prompt_pools[0].total_prompt_length
            else:
                additional_length = 0

            instance_output = torch.index_select(attention_out, 0, torch.tensor(i).cuda())
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i][1], e21[i][1]]).cuda() + additional_length)
            output.append(instance_output)
        output = torch.cat(output, dim=0)
        output = output.view(output.shape[0], -1)
        return output
