# Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation

This repository contains the code and data for the paper **Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation**. 

Our framework evaluates LLM-generated citations by supporting generation with configurable citation granularities ($k$) and volume-controlled stratification. It includes an automated LLM evaluation pipeline for scoring attribution quality (**Citation Precision**, **Citation Recall**, **Citation F1**) and **Answer Correctness**.

> **Acknowledgment:** This codebase is heavily adapted from the [LongBench-Cite repository](https://github.com/THUDM/LongBench-Cite). We extend their foundational evaluation framework to support variable citation chunking (granularity control) and volume-controlled analysis.

## 1. Setup Environment

First, make sure you have installed the required dependencies.

```bash
pip install -r requirements.txt
```
---

## 2. Start vLLM Server

Since generating predictions and evaluating metric models like Qwen requires an LLM inference backend, we rely on `vLLM` internally.

To start an LLM server, you can use the provided bash script.

```bash
chmod +x start_vllm.sh
./start_vllm.sh meta-llama/Llama-3.1-8B-Instruct 4 8000
```

By default, it provisions the server at `http://localhost:8000/v1`. See `start_vllm.sh` to customize your GPU settings, context length boundaries, or adjust for Singularity/Docker deployments if computing on HPC clusters.

> **Note:** Ensure your environment variables (`VLLM_API_URL` and `VLLM_API_KEY`) are exported in your terminal to align with your server's configuration before running the prediction scripts.

---

## 3. Generate Predictions

You can run `predict.py` to prompt an LLM to generate answers with citation markers across the LongBench-Cite dataset. This script utilizes our **Unified Granularity Prompting** strategy, allowing you to manipulate how the source document is chunked.

```bash
python predict.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "data/LongBench-Cite_en.json" \
    --n_sentences 4 \
    --output_dir "predictions" \
    --num_workers 4
```

### Arguments:
- `--model`: The model identifier to hit your OpenAI/vLLM endpoint.
- `--dataset`: Path to the json formatted evaluation dataset.
- `--n_sentences`: The citation granularity ($k$); the number of consecutive sentences per text chunk when segmenting the source document context (e.g., 1, 2, 4, 8, 16, 32).
- `--output_dir`: Directory to save final predictions.
- `--num_workers`: Threads used for asynchronous parallel generation.

---

## 4. Evaluator: Attribution Quality

Once predictions are ready, evaluate the model's **Citation Precision**, **Citation Recall**, and **Citation F1** using an external LLM judge. Following our paper, we recommend using `Qwen3-Next-80B-A3B-Instruct` for maximum reproducibility.

```bash
python evaluate_citation.py \
    --pred_dir "predictions/Llama-3.1-8B-Instruct" \
    --output_dir "scores_cite" \
    --num_workers 4
```

### Arguments:
- `--pred_dir`: The directory path containing the `.json` predictions generated from step 3.
- `--output_dir`: Path to save the evaluation score json.
- `--num_workers`: Threads used for parallel requests.

---

## 5. Evaluator: Answer Correctness

We evaluate the overall factual correctness and comprehensiveness of the model's complete answer by comparing it against the ground-truth reference answers provided in the dataset.

```bash
python evaluate_correctness.py \
    --pred_dir "predictions/Llama-3.1-8B-Instruct" \
    --output_dir "scores_correct" \
    --num_workers 4
```

Similar to citation evaluation, the result `.json` file containing dataset grouped metrics will be deposited into the designated output directory.

---

## 6. Volume-Controlled Stratification

To adhere to the **Volume-Controlled Stratification** framework formalized in Section 3.3 of the primary paper, we isolate citation volume ($V$) from granularity ($k$). Without this control, coarser granularities (e.g., $k=32$) would naturally encompass substantially more text than fine-grained ones ($k=1$), creating an imbalance in the evidence available for grounding and making direct comparisons invalid.

The `evaluate_stratification.py` script automatically groups the raw metrics generated in Step 4 into discrete volume bins and generates performance CSV sheets.

```bash
python evaluate_stratification.py \
    --eval_dir "scores_cite" \
    --output_dir "stratification_results"
```

This will deposit `bucket_performance.csv` containing the aggregate Model-Bucket-Granularity averages to easily import into standard analytical environments.
