# Cloned Repositories

This directory contains code repositories relevant to dialogue agent memory gap detection research.

## Repository 1: LangChain Memory Agent

- **URL**: https://github.com/langchain-ai/memory-agent
- **Purpose**: Reference implementation for conversational memory in LangChain agents
- **Location**: code/memory-agent/
- **Key Features**:
  - ReAct-style agent with memory tool
  - Persistent user-scoped memories across sessions
  - Memory storage and retrieval mechanisms
  - Integration with LangGraph

### Key Files
- `src/agent/` - Agent implementation with memory
- `src/memory/` - Memory management system
- `README.md` - Setup and usage instructions

### How to Use
This implementation demonstrates how to build agents that can:
1. Save important information to long-term memory
2. Retrieve relevant memories during conversations
3. Scope memories to specific users
4. Persist memories across conversational threads

**Relevance to Research**: Provides baseline implementation for memory management that can be extended to detect memory gaps.

---

## Repository 2: IC-DST (In-Context Learning for Dialogue State Tracking)

- **URL**: https://github.com/Yushi-Hu/IC-DST
- **Purpose**: State-of-the-art dialogue state tracking baseline
- **Location**: code/IC-DST/
- **Paper**: EMNLP 2022
- **Key Features**:
  - Few-shot dialogue state tracking
  - In-context learning approach
  - MultiWOZ 2.1 evaluation
  - Prompt-based DST

### Key Files
- `create_data.py` - Data preprocessing scripts
- `run.py` - Main training/evaluation script
- `utils/` - Helper functions for DST
- `prompts/` - Prompt templates

### Key Metrics
- Joint Goal Accuracy on MultiWOZ
- Slot accuracy per domain

### How to Use
```bash
# Install requirements
pip install -r requirements.txt

# Preprocess data
python create_data.py

# Run evaluation
python run.py --model_name gpt-3.5-turbo
```

**Relevance to Research**: Dialogue state tracking requires maintaining conversation state (memory). Errors in DST often indicate memory gaps - perfect for testing memory gap detection.

---

## Repository 3: LLM-Uncertainty-Bench

- **URL**: https://github.com/smartyfh/LLM-Uncertainty-Bench
- **Purpose**: Comprehensive benchmark for uncertainty quantification in LLMs
- **Location**: code/LLM-Uncertainty-Bench/
- **Key Features**:
  - Multiple uncertainty estimation methods
  - Benchmarks across diverse tasks
  - Includes dialogue response selection task
  - Evaluation metrics for uncertainty calibration

### Key Files
- `src/uncertainty_estimation/` - Uncertainty quantification methods
- `src/calibration/` - Calibration metrics
- `datasets/` - Dataset loaders
- `evaluation/` - Evaluation scripts

### Uncertainty Methods Implemented
1. **Verbalized Confidence**: Ask model to express confidence
2. **Token Probability**: Use softmax probabilities
3. **Consistency-based**: Sample multiple responses and measure consistency
4. **Ensemble methods**: Aggregate predictions from multiple models

### How to Use
```bash
# Install dependencies
pip install -r requirements.txt

# Run uncertainty estimation
python run_uncertainty.py --task dialogue_response_selection

# Evaluate calibration
python evaluate_calibration.py
```

**Relevance to Research**: Directly applicable to measuring response divergence across different memory policies. The uncertainty methods can be adapted to detect when memory gaps cause inconsistent responses.

---

## Repository 4: Awesome-LLM-Uncertainty-Reliability-Robustness

- **URL**: https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness
- **Purpose**: Curated list of resources on LLM uncertainty
- **Location**: code/awesome-llm-uncertainty/
- **Key Features**:
  - Comprehensive paper collection
  - Organized by topic and method
  - Links to code implementations
  - Regular updates

### Main Categories
1. **Uncertainty Quantification**
   - Confidence estimation
   - Calibration methods
   - Selective prediction

2. **Reliability**
   - Consistency evaluation
   - Factuality checking
   - Hallucination detection

3. **Robustness**
   - Out-of-distribution detection
   - Adversarial robustness
   - Domain adaptation

### How to Use
Browse the README.md for:
- Recent papers on uncertainty in LLMs
- Code implementations of various methods
- Benchmarks and datasets
- Tools and libraries

**Relevance to Research**: Comprehensive resource for understanding state-of-the-art in uncertainty estimation, which is crucial for detecting when agents are uncertain due to memory gaps.

---

## Integration Recommendations

### For Implementing Memory Gap Detection:

1. **Use memory-agent** as the base framework for building conversational agents with memory

2. **Adapt IC-DST** to track dialogue state and detect when state tracking fails (indicating potential memory gaps)

3. **Apply LLM-Uncertainty-Bench methods** to:
   - Measure response uncertainty when memory is available vs. unavailable
   - Calibrate confidence scores for memory-dependent responses
   - Evaluate consistency across different memory policies

4. **Reference awesome-llm-uncertainty** for:
   - Latest methods in uncertainty quantification
   - Alternative approaches to try
   - Evaluation best practices

### Suggested Workflow:

```python
# 1. Build agent with memory (from memory-agent)
from memory_agent import MemoryAgent

agent = MemoryAgent()

# 2. Track dialogue state (inspired by IC-DST)
state_tracker = DialogueStateTracker()

# 3. Simulate multiple memory policies
responses = []
for policy in [full_memory, partial_memory, no_memory]:
    agent.set_memory_policy(policy)
    response = agent.generate_response(user_input)
    responses.append(response)

# 4. Measure divergence (using LLM-Uncertainty-Bench methods)
from uncertainty_bench import measure_consistency

divergence_score = measure_consistency(responses)

# 5. Detect memory gap
if divergence_score > threshold:
    print("Memory gap detected!")
    agent.take_corrective_action()
```

---

## Dependencies and Setup

### Common Requirements:
- Python 3.8+
- PyTorch 1.13+
- Transformers 4.30+
- LangChain (for memory-agent)
- OpenAI API key (for LLM-based methods)

### Installation:
Each repository has its own `requirements.txt`. Install dependencies for each as needed:

```bash
cd memory-agent && pip install -r requirements.txt
cd ../IC-DST && pip install -r requirements.txt
cd ../LLM-Uncertainty-Bench && pip install -r requirements.txt
```

---

## Additional Notes

### Not Included (but could be added):
- Google's Schema-Guided DST baseline
- TRADE implementation for DST
- Additional memory frameworks (MemGPT, etc.)

### Future Extensions:
- Add implementations of specific papers from the literature review
- Implement custom memory policies for experimentation
- Create evaluation harness combining all tools

---

## Quick Start for Research

1. **Familiarize with memory-agent**: Understand how memory storage/retrieval works
2. **Study IC-DST**: Learn how dialogue state tracking detects conversation state
3. **Experiment with LLM-Uncertainty-Bench**: Test uncertainty methods on sample dialogues
4. **Design experiments**: Combine insights from all three repos to implement memory gap detection

The combination of these repositories provides:
- ✓ Memory management infrastructure
- ✓ Dialogue understanding and state tracking
- ✓ Uncertainty quantification methods
- ✓ Comprehensive background literature

This forms a solid foundation for implementing and evaluating the research hypothesis.
