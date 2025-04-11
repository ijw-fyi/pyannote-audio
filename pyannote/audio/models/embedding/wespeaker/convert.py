# MIT License
#
# Copyright (c) 2023-2024 CNRS
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
from typing import Type

import pytorch_lightning as pl
import torch

import pyannote.audio.models.embedding.wespeaker as wespeaker
from pyannote.audio import Model
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.embedding.wespeaker import BaseWeSpeakerResNet
from pyannote.audio.models.embedding.wespeaker.samresnet import SimAMResNet, SimAMBasicBlock, SimAMResNet34, SimAMResNet100

# --- Configuration ---
# Get checkpoint directory from command line argument
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} /path/to/wespeaker_model_dir")
    sys.exit(1)
wespeaker_checkpoint_dir = sys.argv[1]
print(f"Processing directory: {wespeaker_checkpoint_dir}")

wespeaker_checkpoint = Path(wespeaker_checkpoint_dir) / "wespeaker.pt"
if not wespeaker_checkpoint.is_file():
    print(f"Error: wespeaker.pt not found in {wespeaker_checkpoint_dir}")
    sys.exit(1)

pyannote_checkpoint = Path(wespeaker_checkpoint_dir) / "pytorch_model.bin"

# --- Model Identification ---
dir_name = Path(wespeaker_checkpoint_dir).name
print(f"Parsing directory name: {dir_name}")

is_simam = "simam" in dir_name.lower()
Klass: Type[BaseWeSpeakerResNet]

if is_simam:
    # SimAM ResNet Models (e.g., simam-resnet34, simam_resnet100)
    pattern = r"simam[_-]?resnet(\d+)"
    match = re.search(pattern, dir_name.lower())
    if not match:
        print(f"Error: Could not determine SimAM ResNet depth from directory name: {dir_name}")
        sys.exit(1)

    depth = match.group(1)
    print(f"Detected SimAM ResNet model with depth: {depth}")

    # Dynamically get the pyannote.audio class (WeSpeakerSimAMResNet34 or WeSpeakerSimAMResNet100)
    class_name = f"WeSpeakerSimAMResNet{depth}"
    try:
        Klass = getattr(wespeaker, class_name)
    except AttributeError:
        print(f"Error: pyannote.audio class '{class_name}' not found.")
        sys.exit(1)

else:
    # Regular ResNet Models (e.g., resnet34, resnet152)
    pattern = r"resnet(\d+)"
    match = re.search(pattern, dir_name.lower())
    if not match:
        print(f"Error: Could not determine ResNet depth from directory name: {dir_name}")
        sys.exit(1)

    depth = match.group(1)
    print(f"Detected standard ResNet model with depth: {depth}")

    # Dynamically get the pyannote.audio class (WeSpeakerResNet34, WeSpeakerResNet152, etc.)
    class_name = f"WeSpeakerResNet{depth}"
    try:
        Klass = getattr(wespeaker, class_name)
    except AttributeError:
        print(f"Error: pyannote.audio class '{class_name}' not found.")
        sys.exit(1)

print(f"Using pyannote.audio model class: {Klass.__name__}")

# --- Model Initialization and Specification ---
# Duration doesn't impact the embedding model itself, only how it might be used in a pipeline
duration = 5.0
specifications = Specifications(
    problem=Problem.REPRESENTATION, resolution=Resolution.CHUNK, duration=duration
)

# Instantiate the target pyannote.audio model
model = Klass()
print("Initialized pyannote.audio model instance.")

# --- Load and Map State Dictionary ---
print(f"Loading WeSpeaker state dict from: {wespeaker_checkpoint}")
original_state_dict = torch.load(wespeaker_checkpoint, map_location=torch.device("cpu"))

mapped_state_dict = {}
unmapped_keys = list(original_state_dict.keys()) # Keep track of keys

print("Mapping state dict keys...")
if is_simam:
    # SimAM models have 'front', 'pooling', 'bottleneck'
    for key, value in original_state_dict.items():
        new_key = None
        if key.startswith("front."):
            new_key = "resnet." + key # Map front.X -> resnet.front.X
        elif key.startswith("pooling."):
            new_key = "resnet." + key # Map pooling.X -> resnet.pooling.X
        elif key.startswith("bottleneck."):
            new_key = "resnet." + key # Map bottleneck.X -> resnet.bottleneck.X

        if new_key:
            print(f"  Mapping '{key}' -> '{new_key}'")
            mapped_state_dict[new_key] = value
            if key in unmapped_keys:
                unmapped_keys.remove(key)
        else:
            print(f"  Warning: Key '{key}' not mapped for SimAM model.")

else:
    # Standard ResNet models might have a 'projection' layer to remove
    # and weights are directly loaded into model.resnet
    if "projection.weight" in original_state_dict:
        print("  Removing 'projection.weight' from state dict.")
        original_state_dict.pop("projection.weight")
        if "projection.weight" in unmapped_keys:
            unmapped_keys.remove("projection.weight")

    # For standard ResNets, the keys should directly match model.resnet's state_dict
    mapped_state_dict = original_state_dict
    # We assume all remaining keys are for resnet
    unmapped_keys = []
    print("  Standard ResNet: Assuming direct mapping to model.resnet.")


if unmapped_keys:
    print("\nWarning: The following keys from the original state dict were not mapped:")
    for key in unmapped_keys:
        print(f"  - {key}")
    print("This might indicate an incompatibility or an incomplete mapping.\n")

# --- Load Mapped Weights into Pyannote Model ---
print("Loading mapped state dict into pyannote.audio model...")
try:
    if is_simam:
        # Load into the top-level model for SimAM structure
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    else:
        # Load directly into the 'resnet' submodule for standard ResNet structure
        missing_keys, unexpected_keys = model.resnet.load_state_dict(mapped_state_dict, strict=False)

    if missing_keys:
        print("\nWarning: Missing keys when loading state dict:")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print("\nWarning: Unexpected keys when loading state dict:")
        for key in unexpected_keys:
            print(f"  - {key}")
    if not missing_keys and not unexpected_keys:
        print("State dict loaded successfully (strict=False).")
    else:
        print("State dict loaded with warnings (strict=False).")

except Exception as e:
    print(f"\nError loading state dict: {e}")
    print("Conversion failed.")
    sys.exit(1)

# --- Finalize and Save Pyannote Checkpoint ---
model.specifications = specifications

print("Creating PyTorch Lightning checkpoint...")
checkpoint = {"state_dict": model.state_dict()}
model.on_save_checkpoint(checkpoint) # Allow model to add hyperparameters etc.
checkpoint["pytorch-lightning_version"] = pl.__version__

print(f"Saving pyannote.audio checkpoint to: {pyannote_checkpoint}")
torch.save(checkpoint, pyannote_checkpoint)

print("Conversion complete.")

# --- Verification (Optional) ---
try:
    print("\nVerifying by loading the saved model...")
    loaded_model = Model.from_pretrained(pyannote_checkpoint)
    print("Successfully loaded the converted model:")
    print(loaded_model)
except Exception as e:
    print(f"\nError verifying the saved model: {e}")
