# Downloaded Datasets

This directory contains datasets for the research project on teaching dialogue agents to detect their own memory gaps. Data files are NOT committed to git due to size. Follow the download instructions below.

## Quick Start

```bash
# Download HuggingFace datasets
python3 download_hf_datasets.py

# Clone GitHub repositories with datasets
git clone https://github.com/snap-research/locomo.git
git clone https://github.com/yinzhangyue/SelfAware.git
git clone https://github.com/budzianowski/multiwoz.git

# Validate all downloads
python3 validate_datasets.py
```

## Dataset 1: LoCoMo (Long-term Conversational Memory)

### Overview
- **Source**: https://github.com/snap-research/locomo
- **Paper**: https://arxiv.org/abs/2402.17753
- **Size**: 10 conversations, 17MB
- **Format**: JSON
- **Task**: Long-term conversational memory evaluation
- **Splits**: 10 very long conversations (300 turns avg, 9K tokens, up to 35 sessions each)
- **License**: Check repository

### Why This Dataset?
Perfect for testing memory gap detection across extended multi-session dialogues. Each conversation is annotated for question-answering, event summarization, and multi-modal dialogue generation tasks.

### Download Instructions

```bash
git clone https://github.com/snap-research/locomo.git
```

The main dataset is at `locomo/data/locomo10.json`.

### Loading the Dataset

```python
import json

with open('locomo/data/locomo10.json', 'r') as f:
    locomo_data = json.load(f)

print(f"Number of conversations: {len(locomo_data)}")
# Each conversation has multiple sessions and turns
```

### Sample Data Structure

```json
{
  "conversation_id": "...",
  "sessions": [...],
  "turns": [...],
  "qa_annotations": [...],
  "event_summaries": [...]
}
```

### Notes
- Images are not included; web URLs, captions and search queries for images are provided
- Excellent for testing memory retrieval across long time spans
- Tests both factual recall and temporal reasoning

---

## Dataset 2: SelfAware (Unanswerable Questions)

### Overview
- **Source**: https://github.com/yinzhangyue/SelfAware
- **Paper**: https://arxiv.org/abs/2305.18153 (ACL 2023 Findings)
- **Size**: 3,369 questions, 1.4MB
- **Format**: JSON
- **Task**: Self-knowledge evaluation, uncertainty detection
- **Splits**: 2,337 answerable + 1,032 unanswerable questions
- **License**: CC-BY-SA-4.0

### Why This Dataset?
Directly tests an agent's ability to recognize what it doesn't know - a core component of memory gap detection. Questions span five categories of unanswerability.

### Download Instructions

```bash
git clone https://github.com/yinzhangyue/SelfAware.git
```

The dataset is at `SelfAware/data/SelfAware.json`.

### Loading the Dataset

```python
import json

with open('SelfAware/data/SelfAware.json', 'r') as f:
    selfaware = json.load(f)

# Access examples
examples = selfaware['example']
print(f"Total questions: {selfaware['quantity']}")
```

### Question Categories

Unanswerable questions include:
1. **No scientific consensus**: Questions science hasn't definitively answered
2. **Imagination**: Hypothetical scenarios
3. **Completely subjective**: Personal preference questions
4. **Too many variables**: Questions with indefinite answers
5. **Philosophical**: Abstract philosophical questions

Answerable questions sourced from: SQuAD, HotpotQA, TriviaQA

### Sample Data

```json
{
  "question_id": 1,
  "question": "What causes consciousness?",
  "answerable": "unanswerable",
  "category": "philosophical"
}
```

### Notes
- Critical for training/testing uncertainty estimation
- Benchmarks model's self-knowledge capabilities
- Use for measuring when agents should express uncertainty

---

## Dataset 3: MultiWOZ (Task-Oriented Dialogue State Tracking)

### Overview
- **Source**: https://github.com/budzianowski/multiwoz
- **Paper**: MultiWOZ 2.1-2.4 (various versions)
- **Size**: 10,000+ dialogues, 199MB
- **Format**: JSON
- **Task**: Dialogue state tracking across multiple domains
- **Splits**: Train, validation, test (1K each for val/test)
- **License**: Apache 2.0

### Why This Dataset?
Task-oriented dialogues require maintaining dialogue state (memory) across turns. Perfect for testing whether agents can detect when they've lost track of the conversation state (memory gap).

### Download Instructions

**Option 1: Clone repository**
```bash
git clone https://github.com/budzianowski/multiwoz.git
```

**Option 2: Download specific version from HuggingFace**
Note: The HuggingFace dataset script is deprecated. Use the GitHub repository instead.

### Loading the Dataset

```python
import json

# Load from cloned repository
with open('multiwoz/data/MultiWOZ_2.1/data.json', 'r') as f:
    multiwoz = json.load(f)
```

### Dataset Statistics
- 10,438 dialogues total
- 3,406 single-domain dialogues
- 7,032 multi-domain dialogues (2-5 domains)
- 8 domains: Restaurant, Hotel, Attraction, Train, Taxi, Hospital, Police, Bus

### Sample Data

Each dialogue contains:
- Multiple turns with user utterances and system responses
- Dialogue state annotations (slot-value pairs)
- Domain labels
- Dialogue acts

### Notes
- Multiple versions (2.1, 2.2, 2.4) with improved annotations
- Well-established benchmark for dialogue state tracking
- Good for testing memory consistency across domain switches

---

## Dataset 4: Prosocial Dialog (Multi-turn Conversations)

### Overview
- **Source**: https://huggingface.co/datasets/allenai/prosocial-dialog
- **Paper**: ProsocialDialog paper (Allen AI)
- **Size**: 165,681 total examples, 50MB
- **Format**: HuggingFace Dataset
- **Task**: Multi-turn dialogue with prosocial responses
- **Splits**: train (120,236), validation (20,416), test (25,029)
- **License**: Apache 2.0

### Why This Dataset?
Large-scale multi-turn dialogue dataset (58K dialogues, 331K utterances) ideal for training conversational agents and testing context retention.

### Download Instructions

**Using the provided script:**
```bash
python3 download_hf_datasets.py
```

**Or manually:**
```python
from datasets import load_dataset

prosocial = load_dataset("allenai/prosocial-dialog")
prosocial.save_to_disk("prosocial_dialog")
```

### Loading the Dataset

Once downloaded:
```python
from datasets import load_from_disk

dataset = load_from_disk("prosocial_dialog")
print(dataset['train'][0])
```

### Data Fields

Each example contains:
- `context`: Previous dialogue turns
- `response`: System response
- `rots`: Rules of thumb (social norms)
- `safety_label`: Safety classification
- `safety_annotations`: Detailed safety annotations

### Sample Data

```python
{
  'context': 'I saw my neighbor struggling with groceries.',
  'response': 'It\'s kind to offer help to neighbors in need.',
  'rots': ['Help others when you can'],
  'safety_label': '__ok__',
  'safety_annotations': {...}
}
```

### Notes
- First large-scale multi-turn prosocial dialogue dataset
- Good for testing conversational coherence
- Includes social reasoning annotations

---

## Dataset Summary Table

| Dataset | Size | Type | Key Feature | Use Case |
|---------|------|------|-------------|----------|
| **LoCoMo** | 10 convos, 17MB | Long-term memory | 300+ turns, 35 sessions | Memory across sessions |
| **SelfAware** | 3.4K questions, 1.4MB | Self-knowledge | Answerable/unanswerable | Uncertainty detection |
| **MultiWOZ** | 10K dialogues, 199MB | Task-oriented | Dialogue state tracking | State consistency |
| **Prosocial Dialog** | 166K examples, 50MB | Multi-turn chat | Social reasoning | Context retention |

---

## Recommended Dataset Usage for Research

### For Memory Gap Detection:
1. **Primary**: LoCoMo - Test detection across very long conversations
2. **Secondary**: MultiWOZ - Test state tracking and memory consistency

### For Uncertainty Estimation:
1. **Primary**: SelfAware - Direct self-knowledge evaluation
2. **Secondary**: All datasets - Measure confidence on domain-specific questions

### For Policy Simulation:
1. Use LoCoMo and MultiWOZ to simulate different memory retention policies
2. Measure response divergence when memory policies differ

### For Baseline Comparisons:
- Use standard splits from each dataset
- Report results on test sets
- Compare with published benchmarks

---

## Validation

After downloading all datasets, run the validation script:

```bash
python3 validate_datasets.py
```

This will check:
- All datasets are present
- Data can be loaded successfully
- Sample counts match expected values
- Data structures are correct

---

## Troubleshooting

### HuggingFace Authentication
Some HuggingFace datasets may require authentication:
```bash
huggingface-cli login
```

### Large Download Times
- LoCoMo: ~1 min
- SelfAware: ~30 sec
- MultiWOZ: ~2 min
- Prosocial Dialog: ~5 min

### Disk Space
Ensure you have at least 500MB free for all datasets.

---

## Citation

If you use these datasets, please cite the original papers:

**LoCoMo:**
```bibtex
@article{maharana2024locomo,
  title={Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author={Maharana, Adyasha and others},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```

**SelfAware:**
```bibtex
@inproceedings{yin2023selfaware,
  title={Do Large Language Models Know What They Don't Know?},
  author={Yin, Zhangyue and others},
  booktitle={ACL 2023 Findings},
  year={2023}
}
```

**MultiWOZ:**
```bibtex
@inproceedings{budzianowski2018multiwoz,
  title={MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
  author={Budzianowski, Paweł and others},
  booktitle={EMNLP},
  year={2018}
}
```

**Prosocial Dialog:**
```bibtex
@inproceedings{kim2022prosocialdialog,
  title={ProsocialDialog: A Prosocial Backbone for Conversational Agents},
  author={Kim, Hyunwoo and others},
  booktitle={EMNLP},
  year={2022}
}
```
