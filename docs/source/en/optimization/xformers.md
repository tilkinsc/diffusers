<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# xFormers

We recommend [xFormers](https://github.com/facebookresearch/xformers) for both inference and training. In our tests, the optimizations performed in the attention blocks allow for both faster speed and reduced memory consumption.

Install xFormers from `pip`:

```bash
# For CPU
pip install xformers
# For CUDA version 11.8
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
# For CUDA version 12.1
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

<Tip>

The xFormers `pip` package requires the latest version of PyTorch. If you need to use a previous version of PyTorch, then we recommend [installing xFormers from the source](https://github.com/facebookresearch/xformers#installing-xformers).

</Tip>

After xFormers is installed, you can use the `enable_xformers_memory_efficient_attention()` function for faster inference and reduced memory consumption as shown in this [section](memory#memory-efficient-attention).

Using xFormers:

```py
from diffusers import DiffusionPipeline
import torch

# Load a model
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda") # Assuming you are using CUDA instead of CPU

# Enable xformers
pipeline.enable_xformers_memory_efficient_attention()
```

<Tip warning={true}>

According to this [issue](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212), xFormers `v0.0.16` cannot be used for training (fine-tune or DreamBooth) in some GPUs. If you observe this problem, please install a development version as indicated in the issue comments.

</Tip>
