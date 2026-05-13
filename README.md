# A Multi-Agent Evaluation Framework for Clinical Mental Health Diagnosis

This repository contains the official implementation of a multi-agent evaluation
framework for clinical mental health diagnosis. The project studies whether
collaborative large language model workflows can improve the reliability,
interpretability, and factual consistency of PHQ-8 depression severity
prediction on the DAIC-WOZ clinical interview dataset.(Link for dataset:https://dcapswoz.ict.usc.edu/)

Conventional single-model zero-shot prompting can identify psychological signals
from conversational text, but it often suffers from hallucinated reasoning, weak
verification, and limited transparency in sensitive diagnostic tasks. This
framework addresses these limitations by decomposing the diagnostic process into
specialized agents implemented with a modular LangGraph workflow.

## Overview

The proposed system simulates a collaborative clinical consultation process with
four specialized agents:

- `Perception Agent`: extracts symptom-relevant behavioral signals from patient
  transcripts and visual action unit summaries.
- `Knowledge Agent`: retrieves psychiatric references with retrieval-augmented
  generation using `nomic-embed-text` embeddings and top-k retrieval.
- `Reasoning Agent`: performs structured PHQ-8 item-level clinical inference and
  produces evidence-grounded severity scores.
- `Audit Agent`: verifies diagnostic consistency, detects unsupported inference,
  and revises predictions when evidence is insufficient.

All models are deployed locally through Ollama to preserve the privacy of
clinical data. The default implementation uses `llama3.1:8b`, `qwen2.5:7b`, and
`deepseek-r1:8b`.

## Contributions

This codebase supports the following research contributions:

- A modular multi-agent framework for clinical mental health diagnosis using
  collaborative LLM workflows.
- An integrated pipeline combining behavioral perception, RAG-based psychiatric
  grounding, structured clinical reasoning, and audit-based verification.
- A reproducible experimental setup for comparing single-agent baselines,
  multi-agent orchestration, and ablated pipeline variants on DAIC-WOZ.
- Process-oriented evaluation metrics for reasoning reliability,
  interpretability, audit correction, hallucination detection, and knowledge
  grounding.

## Repository Structure

```text
.
|-- agents.py              # Core LangGraph agent nodes
|-- graph.py               # Default multi-agent workflow
|-- state.py               # Shared graph state definition
|-- utils.py               # DAIC-WOZ transcript and AU feature loading
|-- data_split.py          # Session-level development/validation/test split
|-- knowledge_base.py      # Psychiatric reference retrieval module
|-- experiment_runner.py   # Single-agent, multi-agent, and ablation experiments
|-- evaluate_results.py    # Metrics, confidence intervals, and t-tests
|-- main.py                # Simple benchmark entry point
`-- README.md
```

## Dataset

The experiments use the DAIC-WOZ clinical interview dataset for PHQ-8 depression
severity prediction. After data purification, 184 validated sessions are split
at the session level into development, validation, and held-out test subsets.
This prevents transcripts from the same participant from appearing across
multiple subsets.

The expected file layout is:

```text
Dataset/
|-- data_split_Depression_AVEC2017.csv
|-- 300_TRANSCRIPT.csv
|-- 300_CLNF_AUs.csv
|-- 301_TRANSCRIPT.csv
|-- 301_CLNF_AUs.csv
`-- ...
```

The dataset is not included in this repository. Users must obtain DAIC-WOZ
through the appropriate data access procedure and configure the dataset path
locally.

## Data Split

Create the development, validation, and held-out test splits after data
purification:

```bash
python data_split.py --seed 42
```

For 184 validated sessions, the script produces:

- `data_splits/development.csv`: 128 sessions
- `data_splits/validation.csv`: 19 sessions
- `data_splits/test.csv`: 37 sessions
- `data_splits/split_manifest.json`: split metadata and counts

The split is generated at the session level and validates that no participant
appears in more than one subset.

## Model Configuration

All experiments use locally deployed Ollama models. The default configuration is:

- Perception Agent: `llama3.1:8b`
- Knowledge Agent retrieval: `nomic-embed-text`, top-k = 3
- Reasoning Agent: `deepseek-r1:8b`
- Audit Agent: `qwen2.5:7b`
- Temperature: `0.2`

The knowledge retrieval module falls back to deterministic lexical matching if
the local Ollama embedding interface is unavailable.

No parameter fine-tuning or gradient-based optimization is performed. All
experiments are inference-only.

## Running Experiments

Run the full multi-agent pipeline on the held-out test set across three seeds:

```bash
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode multi_agent --config full_pipeline --output-dir results/full_pipeline
```

Run single-agent baselines:

```bash
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode single_agent --single-agent-model llama3.1:8b --output-dir results/single_llama31
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode single_agent --single-agent-model qwen2.5:7b --output-dir results/single_qwen25
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode single_agent --single-agent-model deepseek-r1:8b --output-dir results/single_deepseek
```

Run ablation configurations:

```bash
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode multi_agent --config audit_only --output-dir results/audit_only
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode multi_agent --config knowledge_no_rag --output-dir results/knowledge_no_rag
python experiment_runner.py --data-root --split-csv data_splits/test.csv --split-name test --mode multi_agent --config basic_core_only --output-dir results/basic_core_only
```

The ablation settings correspond to the following configurations:

- `full_pipeline`: Perception, Knowledge, Reasoning, and Audit agents.
- `audit_only`: Perception, Reasoning, and Audit agents.
- `knowledge_no_rag`: Perception, static Knowledge, and Reasoning agents.
- `basic_core_only`: Perception and Reasoning agents.

## Evaluation Metrics

Summarize MAE, standard deviation across seeds, bootstrap confidence intervals,
Audit Correction Rate, Hallucination Rate, Knowledge Grounding Consistency,
Reasoning Stability, and Interpretability Quality:

```bash
python evaluate_results.py --inputs results/full_pipeline results/audit_only results/knowledge_no_rag results/basic_core_only results/single_llama31 results/single_qwen25 results/single_deepseek --output-dir results_summary --bootstrap-iterations 1000
```

Outputs:

- `results_summary/metric_summary.csv`
- `results_summary/pairwise_t_tests.csv`

## Expected Results

The paper reports that the proposed multi-agent pipeline improves PHQ-8 severity
prediction compared with conventional single-agent prompting, reducing MAE from
5.35 to 5.02 on the held-out test setting. Ablation studies further indicate that
the Knowledge and Audit modules help mitigate reasoning drift and unsupported
clinical inference.

Exact values may vary with local Ollama versions, model checkpoints, hardware,
and decoding behavior. The final numbers used for reporting should be generated
from `evaluate_results.py` using the fixed split and seeds described above.

## Privacy and Clinical Use

This repository is intended for research on AI-assisted clinical assessment and
model evaluation. It does not provide medical advice and should not be used as a
standalone diagnostic system. All experiments are designed to run locally so that
clinical transcripts and behavioral features do not need to be sent to external
model APIs.
