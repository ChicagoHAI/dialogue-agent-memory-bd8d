# Downloaded Papers

This directory contains 10 research papers relevant to teaching dialogue agents to detect their own memory gaps through policy simulation.

## Papers on Meta-Cognitive Memory Systems

### 1. [Cognitive Architectures for Language Agents (CoALA)](2309.02427_cognitive_architectures.pdf)
- **Authors**: Sumers et al.
- **Year**: 2023
- **arXiv**: 2309.02427
- **Why relevant**: Provides a foundational framework for organizing dialogue agents along three dimensions: information storage (working and long-term memory), action space (internal and external actions), and decision-making procedures. Discusses how LLMs manage internal state via learning and reasoning.

### 2. [A-Mem: Agentic Memory for LLM Agents](2502.12110_amem_agentic_memory.pdf)
- **Authors**: A-Mem Team
- **Year**: 2025
- **arXiv**: 2502.12110
- **Why relevant**: Proposes a novel agentic memory system that can dynamically organize memories following the Zettelkasten method. Addresses the limitation that current memory systems enable basic storage and retrieval but lack sophisticated memory organization.

### 3. [MIRIX: Multi-Agent Memory System for LLM-Based Agents](2507.07957_mirix_multiagent_memory.pdf)
- **Authors**: MIRIX Team
- **Year**: 2025
- **arXiv**: 2507.07957
- **Why relevant**: Operationalizes six coordinated memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) managed by a meta-memory controller. Highly relevant for understanding how agents can manage different memory policies.

## Papers on Long-Term Conversational Memory

### 4. [Evaluating Very Long-Term Conversational Memory of LLM Agents](2402.17753_longterm_conversational_memory.pdf)
- **Authors**: Long-term Memory Team
- **Year**: 2024
- **arXiv**: 2402.17753
- **Why relevant**: Assesses whether conversational agents can sustain a coherent persona and continuous narrative over time. Directly related to memory gap detection in extended dialogues.

### 5. [Toward Conversational Agents with Context and Time Sensitive Long-term Memory](2406.00057_context_time_sensitive_memory.pdf)
- **Authors**: Context-Sensitive Memory Team
- **Year**: 2024
- **arXiv**: 2406.00057
- **Why relevant**: Constructs a dataset and benchmark for conversational agents that tests questions referring to conversational meta-data and ambiguous questions. Useful for evaluating memory gap detection capabilities.

## Papers on Uncertainty and Confidence Estimation

### 6. [Confidence Estimation for LLM-Based Dialogue State Tracking](2409.09629_confidence_estimation_dst.pdf)
- **Authors**: Confidence Estimation Team
- **Year**: 2024
- **arXiv**: 2409.09629
- **Why relevant**: Provides exhaustive exploration of methods for quantifying and leveraging model uncertainty to improve reliability. Evaluates four methods: softmax, raw token scores, verbalized confidences, and combinations. Critical for measuring response divergence.

### 7. [Efficient Uncertainty Estimation with Gaussian Process for Reliable Dialog Response Retrieval](2303.08599_uncertainty_estimation_gp.pdf)
- **Authors**: GP Uncertainty Team
- **Year**: 2023
- **arXiv**: 2303.08599
- **Why relevant**: Proposes an efficient uncertainty calibration framework (GPF-BERT) using Gaussian Process layer and focal loss. Relevant for developing uncertainty metrics in dialogue agents.

## Papers on Self-Knowledge and Self-Awareness

### 8. [Do Large Language Models Know What They Don't Know?](2305.18153_llm_know_what_they_dont_know.pdf)
- **Authors**: SelfAware Team
- **Year**: 2023
- **arXiv**: 2305.18153
- **Why relevant**: Introduces the SelfAware dataset of unanswerable questions and evaluates 20 LLMs' ability to identify unknowable questions. Demonstrates intrinsic capacity for self-knowledge and shows that in-context learning and instruction tuning can enhance this capability. Directly applicable to memory gap detection.

### 9. [Language Models (Mostly) Know What They Know](2207.05221_lm_mostly_know.pdf)
- **Authors**: Kadavath et al.
- **Year**: 2022
- **arXiv**: 2207.05221
- **Why relevant**: Studies whether language models can evaluate the validity of their own claims and predict which questions they will answer correctly. Shows that larger models are well-calibrated when provided in the right format. Foundational work for self-knowledge capabilities.

### 10. [Internal Consistency and Self-Feedback in Large Language Models: A Survey](2407.14507_internal_consistency_survey.pdf)
- **Authors**: Survey Team
- **Year**: 2024
- **arXiv**: 2407.14507
- **Why relevant**: Comprehensive survey of internal consistency mechanisms and self-feedback in LLMs. Provides broad overview of how models can monitor their own outputs and detect inconsistencies, which is related to detecting response divergence across different memory policies.

## Summary Statistics
- **Total papers**: 10
- **Date range**: 2022-2025 (mostly 2023-2024)
- **Total size**: ~14 MB
- **Main topics**: Memory systems (4), Self-knowledge (3), Uncertainty estimation (2), Surveys (1)
