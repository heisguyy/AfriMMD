import math
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers import (
    M2M100Config,
    M2M100Model,
    SiglipVisionConfig,
    SiglipVisionModel
)

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Decoder,
    PreTrainedModel,
    shift_tokens_right,
    M2M100ScaledWordEmbedding
)
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from siglip_nllb_config import SiglipNLLBConfig


class SiglipNLLBModule(nn.Module):
    def __init__(self, config: SiglipNLLBConfig):
        super().__init__()
        self.config = config
        if self.config.m2m100_config.scale_embedding:
            embed_scale = math.sqrt(self.config.m2m100_config.d_model)
        else:
            embed_scale = 1.0
        self.shared = M2M100ScaledWordEmbedding(
            self.config.m2m100_config.vocab_size,
            self.config.m2m100_config.d_model,
            self.config.m2m100_config.pad_token_id,
            embed_scale=embed_scale
        )
        self.encoder = SiglipVisionModel(self.config.siglip_config)
        self.decoder = M2M100Decoder(
            self.config.m2m100_config,
            mbed_tokens=self.shared
        )
        self.connector = nn.Linear(
            self.config.siglip_config.hidden_size,
            self.config.m2m100_config.d_model
        )

    def forward(
        self,
        pixel_values,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[torch.FloatTensor]]
        ] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        interpolate_pos_encoding: bool = False,
    ):
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        batch_size, sequence_length = encoder_outputs[0].shape[:2]
        encoder_attention_mask = torch.ones(
            (batch_size, sequence_length)
        )

        encoder_hidden_states = self.connector(encoder_outputs[0])
        if use_cache is None:
            use_cache = self.config.use_cache
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SiglipNLLBForConditionalGenerationModule(nn.Module):
    def __init__(self, config: SiglipNLLBConfig):
        super().__init__()
        self.config = config
        self.model = SiglipNLLBModule(config=self.config)
        self.lm_head = nn.Linear(
            self.config.m2m100_config.d_model,
            self.model.shared.num_embeddings,
            bias=False
        )
        

    def forward(
        self,
        pixel_values,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[torch.FloatTensor]]
        ] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        interpolate_pos_encoding: bool = False,
    ):
        outputs = self.model(
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            decoder_head_mask,
            cross_attn_head_mask,
            past_key_values,
            decoder_inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            interpolate_pos_encoding,
        )
        lm_logits = self.lm_head(outputs[0])
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output
        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class SiglipNLLBPretrainedModel(PreTrainedModel):
    config_class = SiglipNLLBConfig
    base_model_prefix = "siglipnllb"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "SiglipTextEmbeddings",
        "SiglipEncoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipMultiheadAttentionPoolingHead","M2M100DecoderLayer"
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True


class SiglipNLLBForConditionalGeneration(SiglipNLLBPretrainedModel):
    pass


if __name__ == "__main__":
    SiglipNLLB = FlaxViTBartForConditionalGeneration.from_vit_bart_pretrained('google/vit-base-patch16-224-in21k', 'facebook/bart-large')
    outputs = SiglipNLLB(pixel_values, input_ids, attention_mask, position_ids, output_hidden_states=True)