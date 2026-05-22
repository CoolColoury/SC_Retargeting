import torch

from transformers import AutoConfig

from base_converter import Converter


class RandomConverter(Converter):
    """Generate random memory vectors directly in the target decoder space."""

    def __init__(
        self,
        src_model_path,
        tgt_model_path,
        common_vocab=None,
        converter_type="random",
        random_seed=42,
    ):
        super().__init__(src_model_path, tgt_model_path, common_vocab, converter_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tgt_config = AutoConfig.from_pretrained(tgt_model_path)
        self.tgt_hidden_size = tgt_config.hidden_size
        self.initializer_range = getattr(tgt_config, "initializer_range", 0.02)

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(random_seed)

    def convert(self, embedding):
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        target_shape = (*embedding.shape[:-1], self.tgt_hidden_size)
        random_memory = torch.empty(
            target_shape,
            dtype=torch.float32,
            device=self.device,
        )
        random_memory.normal_(
            mean=0.0,
            std=self.initializer_range,
            generator=self.generator,
        )
        return random_memory
