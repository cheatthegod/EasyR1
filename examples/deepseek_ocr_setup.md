# DeepSeek-OCR GRPO End-to-End Runbook

This guide walks through setting up EasyR1 from a clean machine and launching GRPO training for DeepSeek-OCR.

## 1. Prepare the Environment

### Option A: Recommended Docker Image
1. Pull the pre-built EasyR1 image (bundles CUDA, flash-attn, transformers, vLLM):
   ```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
   ```
2. Start a container with GPU and shared IPC:
   ```bash
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
   ```
3. (Inside the container) clone and install EasyR1:
   ```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
   ```

### Option B: Native Installation
1. Ensure Python 3.9+ is available.
2. Install dependencies (transformers>=4.54.0, flash-attn>=2.4.3, vllm>=0.8.3):
   ```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
   ```

> Tip: If Hugging Face access is slow, set `export HF_ENDPOINT=https://hf-mirror.com` before downloading models.

## 2. Download or Mount the Model
- The default training script pulls `deepseek-ai/deepseek-ocr`. Replace `MODEL_PATH` with a local checkpoint if needed:
  ```bash
MODEL_PATH=/path/to/your/deepseek-ocr
  ```

## 3. Launch GRPO Training
Run the provided script (edit `MODEL_PATH` inside if you want a different checkpoint):
```bash
bash examples/deepseek_ocr_grpo.sh
```

To override parameters on the fly (e.g., different GPUs or experiment name):
```bash
MODEL_PATH=/path/to/your/deepseek-ocr \
python3 -m verl.trainer.main \
  config=examples/deepseek_ocr_grpo.yaml \
  worker.actor.model.model_path=${MODEL_PATH} \
  trainer.n_gpus_per_node=4 \
  trainer.experiment_name=deepseek_ocr_custom_run
```

Key settings live in `examples/deepseek_ocr_grpo.yaml` (dataset, prompt template, GRPO/KL knobs, actor rollout, offload, and checkpointing). Adjust them as needed for your hardware or dataset variants.

## 4. (Optional) Resume or Merge Checkpoints
- Enable automatic resume by keeping `trainer.find_last_checkpoint=true` (default).
- Merge an actor checkpoint to Hugging Face format after training:
  ```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/<exp_name>/<global_step>/actor
  ```
