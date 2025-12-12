#!/usr/bin/env python3
"""
Script to download HuggingFace datasets for dialogue agent memory research.
"""

from datasets import load_dataset
import os

# Create directories for datasets
os.makedirs("multiwoz_hf", exist_ok=True)
os.makedirs("prosocial_dialog", exist_ok=True)
os.makedirs("multi_turn", exist_ok=True)

print("Downloading MultiWOZ 2.2 from HuggingFace...")
try:
    # MultiWOZ dataset
    multiwoz = load_dataset("multi_woz_v22")
    multiwoz.save_to_disk("multiwoz_hf")
    print(f"✓ MultiWOZ saved: {len(multiwoz['train'])} train, {len(multiwoz['validation'])} val, {len(multiwoz['test'])} test examples")
except Exception as e:
    print(f"✗ MultiWOZ download failed: {e}")

print("\nDownloading Prosocial Dialog dataset...")
try:
    # Prosocial Dialog - multi-turn dialogue dataset
    prosocial = load_dataset("allenai/prosocial-dialog")
    prosocial.save_to_disk("prosocial_dialog")
    print(f"✓ Prosocial Dialog saved: {len(prosocial['train'])} train, {len(prosocial['validation'])} val, {len(prosocial['test'])} test examples")
except Exception as e:
    print(f"✗ Prosocial Dialog download failed: {e}")

print("\nDownloading Multi-turn dataset...")
try:
    # Multi-turn dataset
    multi_turn = load_dataset("SoftAge-AI/multi-turn_dataset")
    multi_turn.save_to_disk("multi_turn")
    print(f"✓ Multi-turn dataset saved: {len(multi_turn['train'])} examples")
except Exception as e:
    print(f"✗ Multi-turn dataset download failed: {e}")

print("\n✓ All downloads completed!")
