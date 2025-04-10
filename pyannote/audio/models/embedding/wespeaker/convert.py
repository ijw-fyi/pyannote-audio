# MIT License
#
# Copyright (c) 2023 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Script used to convert from WeSpeaker to pyannote.audio

import sys
import re
from pathlib import Path

import pytorch_lightning as pl
import torch

import pyannote.audio.models.embedding.wespeaker as wespeaker
from pyannote.audio import Model
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.embedding.wespeaker import BaseWeSpeakerResNet
from pyannote.audio.models.embedding.wespeaker.samresnet import SimAMResNet, SimAMBasicBlock

wespeaker_checkpoint_dir = sys.argv[1]  # /path/to/wespeaker_cnceleb-resnet34-LM

wespeaker_checkpoint = Path(wespeaker_checkpoint_dir) / "wespeaker.pt"

# Extract model type from directory name
dir_name = Path(wespeaker_checkpoint_dir).parts[-1]

# Check if it's a SimAM model
is_simam = "simam" in dir_name.lower()

if is_simam:
    # Extract the model type (e.g., "simam_resnet100" from the directory name)
    pattern = r"simam[_-]?resnet(\d+)"
    match = re.search(pattern, dir_name.lower())
    if match:
        depth = match.group(1)  # "100"
        if depth == "34":
            Klass = wespeaker.WeSpeakerSimAMResNet34
        elif depth == "100":
            # Create a dummy class to properly initialize the model
            class WeSpeakerSimAMResNet100(BaseWeSpeakerResNet):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    num_blocks = [6, 16, 24, 3]
                    self.resnet = SimAMResNet(
                        SimAMBasicBlock, num_blocks, 
                        feat_dim=self.hparams.num_mel_bins, embed_dim=256, pooling_func="TSTP"
                    )
            Klass = WeSpeakerSimAMResNet100
        else:
            print(f"Unsupported SimAM model depth: {depth}")
            sys.exit(1)
            
        print(f"Using SimAM ResNet{depth} model")
    else:
        print(f"Could not determine SimAM model depth from directory name: {dir_name}")
        sys.exit(1)
else:
    # Regular ResNet model
    pattern = r"resnet(\d+)"
    match = re.search(pattern, dir_name.lower())
    if match:
        depth = match.group(1)  # "34"
        Klass = getattr(wespeaker, f"WeSpeakerResNet{depth}")
    else:
        print(f"Could not determine model depth from directory name: {dir_name}")
        sys.exit(1)

print(f"Using model class: {Klass.__name__}")

duration = 5.0  # whatever
specifications = Specifications(
    problem=Problem.REPRESENTATION, resolution=Resolution.CHUNK, duration=duration
)

state_dict = torch.load(wespeaker_checkpoint, map_location=torch.device("cpu"))

# Initialize the model
model = Klass()

# For SimAM models, handle differently as they have a different structure
if is_simam:
    # SimAM models use 'front' for resnet and 'bottleneck' instead of 'seg_1'/'seg_2'
    # Create a mapped state dict for pyannote's model structure
    mapped_state_dict = {}
    
    # Map front layers to resnet
    for key, value in state_dict.items():
        if key.startswith("front."):
            # Convert from 'front.XX' to 'resnet.XX'
            new_key = "resnet." + key[6:]
            mapped_state_dict[new_key] = value
        elif key == "bottleneck.weight":
            # Map bottleneck to seg_1
            mapped_state_dict["resnet.seg_1.weight"] = value
        elif key == "bottleneck.bias":
            mapped_state_dict["resnet.seg_1.bias"] = value
    
    # Load the mapped state dict
    model.load_state_dict(mapped_state_dict, strict=False)
else:
    # Regular ResNet processing
    # Remove projection layer that isn't part of pyannote's models
    if "projection.weight" in state_dict:
        state_dict.pop("projection.weight")
    
    model.resnet.load_state_dict(state_dict, strict=True)

model.specifications = specifications

checkpoint = {"state_dict": model.state_dict()}
model.on_save_checkpoint(checkpoint)
checkpoint["pytorch-lightning_version"] = pl.__version__

pyannote_checkpoint = Path(wespeaker_checkpoint_dir) / "pytorch_model.bin"
torch.save(checkpoint, pyannote_checkpoint)

model = Model.from_pretrained(pyannote_checkpoint)
print(model)
