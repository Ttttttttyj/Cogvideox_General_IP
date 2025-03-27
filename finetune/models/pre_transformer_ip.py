from diffusers.configuration_utils import ConfigMixin,register_to_config
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from diffusers import CogVideoXTransformer3DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND,is_torch_version,logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import get_2d_sincos_pos_embed,TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class GeneralIP_Pre_TransformerModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).

    Parameters:
        num_attention_heads (int, optional, defaults to 30): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 64): The number of channels in each head.
        in_channels (int, defaults to 16): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 6): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 1920,
        num_layers: int = 6,
        dropout: float = 0.0,
        attention_bias: bool = True,
        patch_size: int = 2,
        sample_width: int = 90,
        sample_height: int = 60,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1920,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        timestep_activation_fn: str = "silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_bias = attention_bias
        self.activation_fn = activation_fn
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.patch_size = patch_size
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.upcast_attention = upcast_attention
        self.norm_type = norm_type
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.flip_sin_to_cos = flip_sin_to_cos
        self.timestep_activation_fn = timestep_activation_fn
        self.inner_dim = attention_head_dim * num_attention_heads
        self.gradient_checkpointing = False
        self.freq_shift = freq_shift

        self.patch_embed_for_ref_image = nn.Conv2d(
                self.in_channels, self.inner_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=True
            ) # [b,16,60,90]->[b,1920,30,45]
        
        # 3.transformers blocks
        self.transformer_blocks_for_pre = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    cross_attention_dim = self.inner_dim,
                    dropout=self.dropout,
                    activation_fn=self.activation_fn,
                    # num_embeds_ada_norm=self.num_embeds_ada_norm,
                    attention_bias=self.attention_bias,
                    upcast_attention=True,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_type = self.norm_type,
                    norm_eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(self.inner_dim, norm_eps, norm_elementwise_affine)

        self.proj_out = nn.Linear(self.inner_dim, self.inner_dim)
        # self.conv_layer = nn.Conv2d(in_channels=self.inner_dim, out_channels=self.inner_dim, kernel_size=1, stride=1,bias=True)
        # nn.init.constant_(self.conv_layer.weight, 0.0)
        # nn.init.constant_(self.conv_layer.bias, 0.0) 


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor, #[b,13*60*90,1920]
        ref_image_hidden_states: torch.Tensor, #[b,1,16,60,90]
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input embeddings
        height = self.sample_height // self.patch_size
        width = self.sample_width // self.patch_size
        first_frm_hidden_states = hidden_states[:,: height*width ,:] #[b,30*45,1920]
        

        ref_image_hidden_states = ref_image_hidden_states.squeeze(1) #[b,16,60,90]
        ref_image_hidden_states = self.patch_embed_for_ref_image(ref_image_hidden_states) #[b,1920,30,45]
        ref_image_hidden_states = ref_image_hidden_states.flatten(2).transpose(1, 2)  # [batch, height x width, channels] [b,30*45,1920]

        # concat_hidden_states = torch.cat((first_frm_hidden_states,ref_image_hidden_states),dim=2) #[b,30*45,1920*2]

        # 2. Transformer blocks
        for i, block in enumerate(self.transformer_blocks_for_pre):

            # concat_hidden_states = torch.cat((first_frm_hidden_states,ref_image_hidden_states[:,0,:,:,:]),dim=1) #[b,32,60,90]
            # height, width = first_frm_hidden_states.shape[-2] // self.patch_size, first_frm_hidden_states.shape[-1] // self.patch_size
            # first_frm_hidden_states = self.pos_embed_for_q[i](first_frm_hidden_states) # [b,16,60,90]->[b, 30*45,1920]
            # concat_hidden_states = self.pos_embed_for_kv[i](concat_hidden_states) # [b,32,60,90]->[b, 30*45,1920]
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                first_frm_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    first_frm_hidden_states,
                    None,
                    ref_image_hidden_states,
                    None,
                    timestep,
                    **ckpt_kwargs,
                )
            else:
                first_frm_hidden_states = block(
                    hidden_states=first_frm_hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=ref_image_hidden_states,
                    encoder_attention_mask=None,
                    timestep=timestep,
                )

        first_frm_hidden_states = self.norm_final(first_frm_hidden_states) 

        first_frm_hidden_states = self.proj_out(first_frm_hidden_states) # [b, 30*45, 1920]
        # batch_size = first_frm_hidden_states.shape[0]
        # first_frm_hidden_states = first_frm_hidden_states.reshape(batch_size, height, width, self.inner_dim).permute(0, 3, 1, 2) # [b, 1920, 30, 45]

        # first_frm_hidden_states = self.conv_layer(first_frm_hidden_states.contiguous()) # [b,1920,30,45]
        # first_frm_hidden_states = first_frm_hidden_states.permute(0,2,3,1).flatten(1,2) # [b,30*45,1920]
        # # 3.Unpatchify
        # first_frm_hidden_states = first_frm_hidden_states.reshape(batch_size, height, width, -1, self.patch_size, self.patch_size) #[b,30,45,16,2,2]
        # first_frm_hidden_states = first_frm_hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        return first_frm_hidden_states

class GeneralIP_CogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        pre_num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
    ):
        super().__init__(
            num_attention_heads = num_attention_heads,
            attention_head_dim = attention_head_dim,
            in_channels = in_channels,
            out_channels =  out_channels,
            flip_sin_to_cos = flip_sin_to_cos,
            freq_shift = freq_shift,
            time_embed_dim = time_embed_dim,
            text_embed_dim = text_embed_dim,
            num_layers = num_layers,
            dropout = dropout,
            attention_bias = attention_bias,
            sample_width = sample_width,
            sample_height = sample_height,
            sample_frames = sample_frames,
            patch_size = patch_size,
            temporal_compression_ratio = temporal_compression_ratio,
            max_text_seq_length = max_text_seq_length,
            activation_fn = activation_fn,
            timestep_activation_fn = timestep_activation_fn,
            norm_elementwise_affine = norm_elementwise_affine,
            norm_eps = norm_eps,
            spatial_interpolation_scale = spatial_interpolation_scale,
            temporal_interpolation_scale = temporal_interpolation_scale,
            use_rotary_positional_embeddings = use_rotary_positional_embeddings,
            use_learned_positional_embeddings = use_learned_positional_embeddings,
            )
        self.pre_num_layers = pre_num_layers
        self.inner_dim = attention_head_dim * num_attention_heads


        self.pre_transformer = GeneralIP_Pre_TransformerModel(
            num_attention_heads = self.config.num_attention_heads,
            attention_head_dim = self.config.attention_head_dim,
            in_channels = self.config.in_channels,
            out_channels = self.inner_dim,
            num_layers = self.pre_num_layers,
            dropout = self.config.dropout,
            attention_bias = self.config.attention_bias,
            patch_size = self.config.patch_size,
            sample_width = self.config.sample_width,
            sample_height = self.config.sample_height,
            activation_fn = self.config.activation_fn,
            norm_type = "layer_norm",
            num_embeds_ada_norm = self.inner_dim,
            norm_elementwise_affine = self.config.norm_elementwise_affine,
            norm_eps = self.config.norm_eps,
            flip_sin_to_cos = self.config.flip_sin_to_cos,
            freq_shift=self.config.freq_shift,
            timestep_activation_fn = self.config.timestep_activation_fn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape # [b,13,16,60,90]
        p = self.config.patch_size # 2

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length] #[b,226,1920]
        hidden_states = hidden_states[:, text_seq_length:] #[b,13*30*45,1920]
        # print(f"ref_image_hidden_states mean:{ref_image_hidden_states.mean()} max:{ref_image_hidden_states.max()} min:{ref_image_hidden_states.min()}")
        # print(f"ref_image_hidden_states var:{ref_image_hidden_states.var()}")

        hidden_states_for_former = hidden_states.clone()
        first_frm_hidden_states = self.pre_transformer(hidden_states_for_former, ref_image_hidden_states,timestep)
        del hidden_states_for_former
        hidden_states[:,:height // p * width // p] = first_frm_hidden_states

        # print(f"origin_fisrt_frm_hidden_states mean:{hidden_states[:,:height // p * width // p].mean()} max:{hidden_states[:,:height // p * width // p].max()} min:{hidden_states[:,:height // p * width // p].min()}")
        # print(f"origin_fisrt_frm_hidden_states var:{hidden_states[:,:height // p * width // p].var()}")
        # print(f"fisrt_frm_hidden_states mean:{first_frm_hidden_states.mean()} max:{first_frm_hidden_states.max()} min:{first_frm_hidden_states.min()}")
        # print(f"fisrt_frm_hidden_states var:{first_frm_hidden_states.var()}")

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # p = self.config.patch_size

        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


    


