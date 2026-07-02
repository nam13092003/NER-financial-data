# Financial NER Fine-tuning Pipeline

Instruction-tuned LLM pipeline for **Named Entity Recognition (NER)**
on financial text, built on top of [Unsloth](https://github.com/unslothai/unsloth) + LoRA.

Uses the **FIRE** (Financial Information and Relation Extraction) dataset
with **13 entity types** covering companies, financial entities, monetary
values, and more.

---

## Project Structure

```
NER_finance/
├── config.py                 # PipelineConfig, TrainingFormat enum
├── data/
│   ├── schemas.py            # EntitySpan, StandardizedDocument
│   ├── adapter.py            # FireFormatAdapter (FIRE JSON parsing)
│   └── registry.py           # DatasetRegistry (multi-file aggregation)
├── prompts/
│   ├── builder.py            # BasePromptBuilder (ABC)
│   ├── causal.py             # Format A: plain text input → JSON
│   ├── instruction.py        # Format B: System / User / Assistant (ChatML)
│   ├── factory.py            # PromptBuilderFactory
│   └── _helpers.py           # Shared text formatters (no side-effects)
├── training/
│   ├── collator.py           # DataCollatorWithLossMask (prompt tokens masked)
│   ├── metrics.py            # NER F1 for post-training evaluation
│   ├── model_factory.py      # UnslothModelFactory (LoRA setup)
│   └── trainer.py            # NativeSafeTrainer (multi-GPU safe)
├── utils/
│   └── span_utils.py         # resolve_span_to_index, normalize_term
├── configs/
│   └── default.yaml          # ← Edit this to configure your experiment
├── train.py                  # Entry point for accelerate launch
├── evaluate.py               # Evaluation entry point (multi-GPU inference)
├── requirements.txt
└── README.md
```

---

## Dataset — FIRE (Financial NER)

The FIRE dataset uses JSON array files with the following schema per record:

```json
{
  "tokens": ["Albertsons", "and", "Rite", "Aid", "..."],
  "entities": [
    {"id": "uuid", "text": "Albertsons", "type": "Company", "start": 0, "end": 1},
    {"id": "uuid", "text": "Rite Aid", "type": "Company", "start": 2, "end": 4}
  ]
}
```

| Split | File | Samples |
|---|---|---|
| Train | `fire_train.json` | 2,117 |
| Dev | `fire_dev.json` | 454 |
| Test | `fire_test.json` | 454 |

---

## Configuration

All experiment settings live in **`configs/default.yaml`**.
No code changes needed between experiments — just edit the YAML.

| Key | Options | Default |
|---|---|---|
| `training.format` | `causal` \| `instruction` | `instruction` |
| `model.name` | Any Unsloth model slug | `Phi-3-mini-4k-instruct-bnb-4bit` |

---

## Running on Kaggle Notebook

### Prerequisites
- Kaggle notebook with **2× GPU** (e.g. 2× T4 or P100)
- FIRE dataset uploaded as a Kaggle data source

### Cell 1 — Install Unsloth and dependencies
```python
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
!pip install pyyaml
```

### Cell 2 — Clone this repository
```python
!git clone https://github.com/YOUR_USERNAME/NER_finance.git /kaggle/working/NER_finance
```
> Replace `YOUR_USERNAME` with your GitHub username.

### Cell 3 — (Optional) Modify the config inline
```python
import yaml

CONFIG_PATH = "/kaggle/working/NER_finance/configs/default.yaml"
MY_CONFIG   = "/kaggle/working/my_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ── Change experiment settings here ──────────────────────────
cfg["training"]["format"]  = "instruction"  # causal | instruction
cfg["model"]["name"]       = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
# ─────────────────────────────────────────────────────────────

# Data files
cfg["data"]["train_files"] = [
    {"path": "/kaggle/input/fire-dataset/fire_train.json"},
]
cfg["data"]["eval_files"] = [
    {"path": "/kaggle/input/fire-dataset/fire_dev.json"},
]
cfg["data"]["test_files"] = [
    {"path": "/kaggle/input/fire-dataset/fire_test.json"},
]

# Fast validation / Early stopping tweaks
cfg["training"]["eval_subset_size"] = 0.2
cfg["training"]["early_stopping_patience"] = 3

with open(MY_CONFIG, "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)

print("Config saved to:", MY_CONFIG)
```

### Cell 4 — Configure accelerate for 2 GPUs
```python
!accelerate config default
```

### Cell 5 — Launch training on 2 GPUs
```python
CONFIG = "/kaggle/working/my_config.yaml"   # or use default.yaml

!accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    /kaggle/working/NER_finance/train.py \
    --config {CONFIG} \
    2>&1 | tee /kaggle/working/training_log.txt
```

> **Note:** The `2>&1 | tee` part captures both stdout and stderr into
> `training_log.txt` while still showing output in the notebook.

### Cell 6 — (Optional) Continue Training

#### Scenario A: Resume interrupted training (Keep Optimizer State)
```python
!accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    /kaggle/working/NER_finance/train.py \
    --config {CONFIG} \
    --resume \
    2>&1 | tee -a /kaggle/working/training_log.txt
```

#### Scenario B: Start a completely new run on a previously trained model
Change the model name to point to your merged model folder:
```yaml
model:
  name: "/kaggle/working/outputs/final_merged_model"
```
Then run the normal training command **WITHOUT** the `--resume` flag.

### Cell 7 — Evaluate the trained model (Test Phase)
```python
CONFIG = "/kaggle/working/my_config.yaml"

!accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    /kaggle/working/NER_finance/evaluate.py \
    --config {CONFIG} \
    --checkpoint /kaggle/working/outputs/final_lora_adapter \
    --batch_size 16 \
    --output /kaggle/working/predictions.jsonl
```

> **Resuming Evaluation:** If the notebook crashes during evaluation,
> just run the exact same command again! The script auto-resumes.

---

## Entity Label Ontology (FIRE)

| Label | Description |
|---|---|
| `Action` | Corporate actions such as mergers, acquisitions, buy-outs |
| `BusinessUnit` | Divisions, departments, or subsidiary units |
| `Company` | Names of corporations, firms, and business organizations |
| `Date` | Temporal expressions including dates, time periods, fiscal quarters |
| `Designation` | Professional titles and roles held by individuals |
| `FinancialEntity` | Financial instruments, indices, stock tickers, market indicators |
| `GeopoliticalEntity` | Countries, states, and political regions in economic context |
| `Location` | Physical places, addresses, and geographic references |
| `Money` | Monetary amounts and currency values |
| `Person` | Names of individuals including executives, analysts |
| `Product` | Goods, services, and branded offerings |
| `Quantity` | Numerical values, percentages, and measurable quantities |
| `Sector` | Industry sectors and market segments |

---

## SOLID Design Notes

| Principle | Implementation |
|---|---|
| **S**ingle Responsibility | `FireFormatAdapter` parses only; `PromptBuilder` formats only |
| **O**pen/Closed | New formats added via new `BasePromptBuilder` subclasses |
| **L**iskov Substitution | `CausalPromptBuilder` and `InstructionPromptBuilder` are fully interchangeable |
| **I**nterface Segregation | `BasePromptBuilder` has one abstract method |
| **D**ependency Inversion | `train.py` depends on `BasePromptBuilder`, not concrete classes |
