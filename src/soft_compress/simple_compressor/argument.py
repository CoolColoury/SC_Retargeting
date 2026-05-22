## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class TrainArguments(TrainingArguments):
    compress_method: str = field(
        default="simple_compressor", metadata={"help": "compressor method: icae or pcc or ..."}
    )
    resume_from_checkpoint: bool = field(
        default=False, metadata={"help": "recovery from ckpt"}
    )
    last_ckpt_dir: str = field(
        default=None, metadata={"help": "recovery ckpt dir"}
    )
    compress_model: str = field(
        default=None, metadata={"help": "path to the embedding model"},
    )
    adapter_model: str = field(
        default=None, metadata={"help": "path to the lora adapter model"}
    )
    decoder_model: str = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "path to the language model"},
    )
    output_dir: str = field(
        default="train/save_dir", metadata={"help": "path to save the model"}
    )
    logging_dir: str = field(
        default="train/log_dir", metadata={"help": "path to save the log"}
    )
    deepspeed: str = field(
        default=None, metadata={"help": "path to the deepspeed config file"},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "gradient accumulation steps"}
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "save strategy"}
    )
    save_total_limit: int = field(
        default=2, metadata={"help": "save total limit"}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "logging steps"}
    )
    logging_first_step: bool = field(
        default=True, metadata={"help": "logging first step"}
    )    
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    save_steps: int = field(
        default=5000, metadata={"help": "save steps"}
    ) 
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "batch size per device"}
    )
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "evaluation strategy"}
    )
    eval_steps: int = field(
        default=2000, metadata={"help": "evaluation steps"}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "batch size per device"}
    )
    dataloader_num_workers: int = field(
        default=32, metadata={"help": "number of workers for dataloader"}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "remove unused columns"}
    )
    segment_length: int = field(
        default=256, metadata={"help": "per length of segment"}
    )
    use_mem_toekn: bool = field(
        default=True, metadata={"help": "whether use mem token"}
    )
    embed_len: int = field(
        default=128, metadata={"help": "length of the embedding"}
    )
    drop_out: float = field(
        default=0.25, metadata={"help": "dropout rate"}
    )
    random_seed: int = field(
        default=42, metadata={"help": "random seed"}
    )
    upload_hf: bool = field(
        default=False, metadata={"help": "whether to upload to huggingface"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Control whether use lora to train compressor and recommend open it when use Llama3-Instruct as compressor."}
    )
    lora_r: int = field(
        default=64, metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "lora dropout"}
    )
    compressor_gradient_checkpoint: bool = field(
        default=False, metadata={"help": "whether to use gradient checkpointing for compressor"}
    )

    
@dataclass
class DataArguments:
    train_data_dir: str = field(
        metadata={"help": "path to the training data"},
    )
    valid_data_dir: str = field(
        metadata={"help": "path to the validation data"},
    )
    train_data_samples: int = field(
        default=None, metadata={"help": "number of training data samples"}
    )
    valid_data_samples: int = field(
        default=None, metadata={"help": "number of validation data samples"}
    )