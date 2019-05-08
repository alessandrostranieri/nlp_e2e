import torch
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.nn.util import get_final_encoder_states
from overrides import overrides


@Seq2VecEncoder.register("transformer_encoder")
class TransformerSeq2VecEncoder(Seq2VecEncoder):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 projection_dim,
                 feedforward_hidden_dim,
                 num_layers,
                 num_attention_heads,
                 stateful: bool = False) -> None:
        super().__init__(stateful)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_2_seq = StackedSelfAttentionEncoder(input_dim=input_dim,
                                                     hidden_dim=hidden_dim,
                                                     projection_dim=projection_dim,
                                                     feedforward_hidden_dim=feedforward_hidden_dim,
                                                     num_layers=num_layers,
                                                     num_attention_heads=num_attention_heads)

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        output = self.seq_2_seq(inputs, None)
        return get_final_encoder_states(output, mask)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_dim
