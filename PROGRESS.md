# Progress

## What this is
This file keeps track of things we tried while implementing the project and improving the implementation.

The overall goal of this codebase is to build some variant of a somewhat standard decoder-only pretrained Transformer model.

## Entry Format
Each entry should have:
- Date
- Short title
- some explanations

## Progress Log

### 2026-03-27 - Add validation-loss sanity check
Goal:
Set up validation loss for the initial decoder-only model so we have a simple way to check whether the training setup and model definition are roughly working.

Tried:
Added validation loss tracking to the initial decoder-only training setup.

For this stage, the validation data is intentionally the same as the training data. In that sense, this "validation loss" is being used more like a train-loss sanity check than a true held-out validation metric.

The reason for doing this is simple: if the model and training pipeline are wired correctly, the loss on this repeated data should become very small, given that the model is relatively large (The model should be able to overfit properly). That gives us an early signal that the model definition is at least roughly correct.



