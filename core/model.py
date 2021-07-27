# GPT2 model file, adjust transformers GPT2LMHEADModel
# We adjust the calculation of loss, we only count the loss of title part

from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """GPT2 model"""

    def __init__(self, config):
        # Initialization. configs:parameter configuration
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, past=None, token_type_ids=None, labels=None, title_id=None):
        """
        forward function, calculate GPT2 prediction
        :param input_ids: input sequence embedding in terms of vocabulary, size:[batch_size, sequence_length]
        :param past:contain pre-calculated hidden state of model.
               generally use on prediction state to accelerate decoding and prevent recalculating token counted
        :param token_type_ids: to distinguish between content and title sequence, size:[batch_size, sequence_length]
        :param label: label sequence, size:[batch_size, sequence_length], normally it has same length with input_ids
        :param title_id: title token id
        :return:
        """
        # gain the output of GPT2 model
        transformer_outputs = self.transformer(input_ids, past_key_values=past, token_type_ids=token_type_ids)
        # gain the last hidden state of GPT2 model, size:[batch_size, sequence_length, config.vocab_size]
        hidden_states = transformer_outputs[0]
        # predict next token of last hidden state, size:[batch_size, sequence_length, config.vocab_size]
        lm_logits = self.lm_head(hidden_states)
        # concatenate the result
        outputs = (lm_logits,) + transformer_outputs[1:]
        # if label is not None, then calculate the loss and concatenate the result
        if labels is not None:
            # when calculating the loss, title_id cannot be None since we need it to find title part
            if title_id is None or token_type_ids is None:
                raise Exception("When label is not None, title_id and token_type_ids cannot be None")
            # get the mask: if the parts of token_type_ids equals to title_id, they are needed to be counted in loss,
            # and mark them as 1, 0 otherwise.
            # size:[batch_size, sequence_length]
            mask = (token_type_ids == title_id).long()
            # update the label, size:[batch_size, sequence length]
            labels = labels * mask
            # Shift the prediction outcome and label
            # GPT2 generating mechanism is using all the words ahead to predict the next word
            # Thus, the prediction of first word actually is the second word of label, and so on
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # define loss function: CrossEntropyLoss, set ignoring part of loss, define format of return
            # We will ignore the loss whose shift_labels is 0, which means that we only calculate title part's loss
            # way of calculating loss is sum because we only calculate title part's loss
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # gain the length of title part, and use it to calculate the loss
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs

        return outputs  # (loss),lm_logits, presents, (all hidden_states), (attentions)
