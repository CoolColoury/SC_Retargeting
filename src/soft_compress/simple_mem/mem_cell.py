'''
This code is borrowed from the https://github.com/yurakuratov/hidden_capacity/tree/main repository.
'''

import torch

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, memory_dim=None):
        super().__init__()
        self.model = base_model
        if memory_dim is None:
            memory_dim = base_model.config.hidden_size
        self.memory_dim = memory_dim
        self.num_mem_tokens = num_mem_tokens
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        self.create_memory()

    def create_memory(self):
        embeddings = self.model.get_input_embeddings()
        # Get the device of the embeddings to ensure memory is on the same device
        device = embeddings.weight.data.device
        dtype = embeddings.weight.data.dtype
        std = embeddings.weight.data.std().item()
        # memory_params = torch.randn((self.num_mem_tokens, self.memory_dim), device=device, dtype=dtype) * std
        
        # init memory with 0
        # memory_params = torch.zeros((self.num_mem_tokens, self.memory_dim), device=device, dtype=dtype)

        # init memory with the mean token embeddings
        # mean_token_embeddings = embeddings.weight.data.mean(dim=0).to(device, dtype)
        token_embeddings = embeddings.weight.data.mean(dim=0).to(device, dtype)
        # Try to find a suitable common token for initialization
        # Priority: 1) eos_token (most universal special token), 2) bos_token, 3) pad_token, 4) mean embeddings
        # init_method = None
        
        # Method 1: Try eos_token (more universal than pad_token, exists in most models)
        # eos_token represents "end of sequence" and is semantically neutral for memory initialization
        # try:
        # eos_token_id = getattr(self.model.config, 'eos_token_id', None)
        # if isinstance(eos_token_id, list):
        #     eos_token_id = eos_token_id[0]
        # if eos_token_id is not None and 0 <= eos_token_id < embeddings.weight.data.shape[0]:
        #     token_embeddings = embeddings.weight.data[eos_token_id]
        # else:
        #     print(f"eos_token_id not found in the model config")
        #     raise ValueError(f"eos_token_id not found in the model config")
            # init_method = "eos_token"
        # except:
        #     pass
        
        # Method 2: Try bos_token
        # if token_embeddings is None:
        #     try:
        #         bos_token_id = getattr(self.model.config, 'bos_token_id', None)
        #         if bos_token_id is not None and 0 <= bos_token_id < embeddings.weight.data.shape[0]:
        #             token_embeddings = embeddings.weight.data[bos_token_id]
        #             init_method = "bos_token"
        #     except:
        #         pass
        
        # # Method 3: Try pad_token (fallback)
        # if token_embeddings is None:
        #     try:
        #         pad_token_id = getattr(self.model.config, 'pad_token_id', None)
        #         if pad_token_id is not None and 0 <= pad_token_id < embeddings.weight.data.shape[0]:
        #             token_embeddings = embeddings.weight.data[pad_token_id]
        #             init_method = "pad_token"
        #     except:
        #         pass
        
        # # Method 4: Fallback to mean embeddings (always works, semantically neutral)
        # if token_embeddings is None:
        #     token_embeddings = embeddings.weight.data.mean(dim=0)
        #     init_method = "mean_embeddings"
        
        # print(f"Memory initialized using: {init_method}")
        memory_params = torch.stack([token_embeddings] * self.num_mem_tokens, dim=0).to(device, dtype)
        self.register_parameter('memory', torch.nn.Parameter(memory_params, requires_grad=True))
        self.read_memory_position = range(self.num_mem_tokens)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        # todo: allow labels to be passed, could be used for masking
        labels = input_ids
        logits = out.logits
        labels = labels.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        out.loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'],
                                  attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        mem_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        mem_kwargs['input_ids'] = None
        mem_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            mem_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        mem_kwargs['output_hidden_states'] = True
        return mem_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens:] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithPast()
            # take read memory here
            memory_state = model_outputs.hidden_states[-1][:, self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state