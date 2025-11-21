#!/bin/bash

set -x

# Replace this with your local checkpoint or Hugging Face hub path.
MODEL_PATH=deepseek-ai/deepseek-ocr

# GRPO training for DeepSeek-OCR on DocVQA-style documents.
python3 -m verl.trainer.main \
    config=examples/deepseek_ocr_grpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=deepseek_ocr_docvqa_grpo \
    trainer.n_gpus_per_node=8
