# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
from peft import LoraConfig, get_peft_model, PeftModel

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_visual_layers: int = 0,
        tune_llm_layers: int = 0,
        lora_config: dict | None = None,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            tune_visual_layers: number of last vision layers to unfreeze (0 = all frozen if tune_visual=False)
            tune_llm_layers: number of last LLM layers to unfreeze (0 = all frozen if tune_llm=False)
            lora_config: configuration for LoRA
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_visual_layers, tune_llm_layers, lora_config)

    def set_trainable_parameters(
        self, 
        tune_llm: bool, 
        tune_visual: bool,
        tune_visual_layers: int = 0,
        tune_llm_layers: int = 0,
        lora_config: dict | None = None,
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        # 1. Base freezing logic
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            if isinstance(self.eagle_model, PeftModel):
                 self.eagle_model.base_model.model.language_model.requires_grad_(False)
            else:
                self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            if isinstance(self.eagle_model, PeftModel):
                self.eagle_model.base_model.model.vision_model.requires_grad_(False)
                self.eagle_model.base_model.model.mlp1.requires_grad_(False)
            else:
                self.eagle_model.vision_model.requires_grad_(False)
                self.eagle_model.mlp1.requires_grad_(False)

        # 2. Partial Finetuning
        if tune_visual_layers > 0 and not tune_visual:
            print(f"Partial finetuning: Unfreezing last {tune_visual_layers} vision layers")
            if isinstance(self.eagle_model, PeftModel):
                vision_model = self.eagle_model.base_model.model.vision_model
            else:
                vision_model = self.eagle_model.vision_model
            
            # Assuming Siglip/CLIP structure with encoder.layers
            if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
                layers = vision_model.encoder.layers
                for layer in layers[-tune_visual_layers:]:
                    layer.requires_grad_(True)
            else:
                print("Warning: Could not find vision encoder layers for partial finetuning")

        if tune_llm_layers > 0 and not tune_llm:
            print(f"Partial finetuning: Unfreezing last {tune_llm_layers} LLM layers")
            if isinstance(self.eagle_model, PeftModel):
                language_model = self.eagle_model.base_model.model.language_model
            else:
                language_model = self.eagle_model.language_model
            
            # Assuming Llama/Qwen structure with model.layers
            if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
                layers = language_model.model.layers
                for layer in layers[-tune_llm_layers:]:
                    layer.requires_grad_(True)
            else:
                 print("Warning: Could not find LLM layers for partial finetuning")

        # 3. LoRA
        if lora_config is not None:
            print("Applying LoRA...")
            peft_config = LoraConfig(**lora_config)
            self.eagle_model = get_peft_model(self.eagle_model, peft_config)
            self.eagle_model.print_trainable_parameters()

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual and tune_visual_layers == 0 and tune_llm_layers == 0 and lora_config is None:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if isinstance(self.eagle_model, PeftModel):
                base_model = self.eagle_model.base_model.model
            else:
                base_model = self.eagle_model

            if base_model.language_model and not self.tune_llm:
                base_model.language_model.eval()
            if base_model.vision_model and not self.tune_visual:
                base_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        # Safely remove image_sizes if it exists (not always present)
        eagle_input.pop("image_sizes", None)

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
