# Research Plan: Teaching Dialogue Agents to Detect Their Own Memory Gaps Through Policy Simulation

## Research Question

**Can dialogue agents equipped with a meta-cognitive framework that simulates responses under multiple memory policies and measures response divergence detect their own memory gaps, enabling them to take corrective actions?**

More specifically:
1. Does response divergence across different memory policies correlate with actual memory gaps?
2. Can this divergence signal be used to trigger appropriate corrective actions?
3. Does this approach outperform baseline uncertainty estimation methods?

## Background and Motivation

### The Problem
Dialogue agents using memory systems inevitably forget conversation details due to storage constraints. Current agents don't know when they've forgotten something important, leading to:
- Incorrect or harmful recommendations (e.g., suggesting peanut dishes to users with nut allergies)
- Contradictions with earlier commitments
- Loss of contextual understanding in long conversations

### Why This Matters
According to the LoCoMo paper (arXiv:2402.17753), state-of-the-art LLMs struggle with conversations exceeding 100 turns, with performance degrading significantly for information mentioned 20-30 turns ago. Yet real-world applications (customer service, personal assistants, therapy bots) require maintaining context across dozens or hundreds of turns spanning multiple sessions.

### The Gap in Current Research
From the literature review, I identified:
- **Memory systems exist** (CoALA, A-Mem, MIRIX) but don't detect their own gaps
- **Self-knowledge methods exist** (SelfAware, "LMs Know What They Know") but aren't memory-specific
- **Uncertainty estimation exists** (Confidence DST, GP-based methods) but doesn't distinguish memory gaps from other uncertainty types
- **No existing work** combines policy simulation with divergence measurement for memory gap detection

### Our Contribution
This research proposes a novel meta-cognitive framework where agents:
1. Simulate responses under multiple memory policies (e.g., full context, partial retrieval, no memory)
2. Measure response divergence across these simulations
3. Use high divergence as a signal that memory is critical but uncertain
4. Take corrective actions (search history, ask for clarification, express uncertainty)

## Hypothesis Decomposition

### Primary Hypothesis
**H1**: Response divergence across memory policies correlates with actual memory gaps (i.e., when the agent lacks information needed to answer correctly).

**Testable Predictions**:
- H1a: Divergence will be higher for questions requiring distant past information vs. recent information
- H1b: Divergence will be higher when correct answer depends on specific details vs. general knowledge
- H1c: Divergence scores will predict answer correctness better than baseline uncertainty methods

### Secondary Hypothesis
**H2**: Agents using gap detection can take appropriate corrective actions that improve task performance.

**Testable Predictions**:
- H2a: When high divergence is detected, searching conversation history improves answer accuracy
- H2b: Abstaining from answering when divergence is high improves precision at the cost of recall
- H2c: The gap detection + action system outperforms naive always-answer or always-search baselines

### Control Hypothesis
**H3**: Gap detection generalizes beyond memory to general self-knowledge.

**Testable Predictions**:
- H3a: The method works on SelfAware dataset (unanswerable vs. answerable questions) without memory context
- H3b: Performance is better on memory-dependent tasks (LoCoMo) than general knowledge tasks (SelfAware)

## Proposed Methodology

### High-Level Approach

**Core Idea**: If an agent's response depends heavily on what memory it has access to, then it's operating in a region of high memory uncertainty. By simulating responses under different memory conditions and measuring how much they diverge, we can detect when memory is critical but uncertain.

**Why This Should Work**:
- If memory doesn't matter (general knowledge question), responses will be similar across policies → low divergence
- If memory matters but is accessible (recent conversation), responses will be consistent → low divergence
- If memory matters but is inaccessible (forgotten details), responses will vary wildly across policies → high divergence

**Theoretical Grounding**:
- Builds on self-consistency decoding (Wang et al., "Self-Consistency Improves Chain of Thought Reasoning")
- Related to epistemic uncertainty (GP-based uncertainty paper, arXiv:2303.08599)
- Inspired by counterfactual reasoning ("what if I remembered differently?")

### Experimental Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Gap Detection System               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Dialogue History + Current Query                    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Policy Simulator (runs N policies in parallel)       │  │
│  │                                                         │  │
│  │  Policy 1: Full Context (all history)                 │  │
│  │  Policy 2: Recent Only (last K turns)                 │  │
│  │  Policy 3: Semantic Retrieval (RAG, top-k relevant)   │  │
│  │  Policy 4: No Memory (current turn only)              │  │
│  │                                                         │  │
│  │  → Generate response under each policy                │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Divergence Measurement                                │  │
│  │                                                         │  │
│  │  - Semantic similarity (embedding-based)               │  │
│  │  - Token overlap (ROUGE, BLEU)                         │  │
│  │  - Answer consistency (same factual content?)          │  │
│  │                                                         │  │
│  │  → Divergence Score: [0, 1]                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Action Selection                                      │  │
│  │                                                         │  │
│  │  if divergence > threshold_high:                       │  │
│  │      action = SEARCH_MEMORY or ASK_CLARIFICATION      │  │
│  │  elif divergence > threshold_medium:                   │  │
│  │      action = EXPRESS_UNCERTAINTY                      │  │
│  │  else:                                                 │  │
│  │      action = ANSWER_CONFIDENTLY                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  Output: Response + Confidence + Action Taken               │
└─────────────────────────────────────────────────────────────┘
```

### Experimental Steps

#### Step 1: Implement Memory Policies (Week 1, Day 1)
**Rationale**: Need diverse policies that span different memory availability scenarios.

**Implementation**:
1. **Policy 1 - Full Context**: Include all conversation history (baseline for "perfect memory")
   - Use case: Short conversations that fit in context window
   - Implementation: Simply concatenate all turns

2. **Policy 2 - Recent Only**: Include only last K turns (K=10 by default)
   - Use case: Recency-biased memory
   - Implementation: Sliding window over conversation

3. **Policy 3 - Semantic Retrieval (RAG)**: Retrieve top-k most relevant turns using embedding similarity
   - Use case: Intelligent selective memory
   - Implementation: Use sentence-transformers (all-MiniLM-L6-v2) for retrieval

4. **Policy 4 - No Memory**: Current turn only
   - Use case: Complete amnesia baseline
   - Implementation: Zero-shot response with no history

**Validation**: Run each policy independently and verify responses make sense.

#### Step 2: Implement Divergence Measurement (Week 1, Day 1-2)
**Rationale**: Need reliable, calibrated divergence scores.

**Methods to Implement**:
1. **Semantic Similarity** (primary metric)
   - Embed each response using sentence-transformers
   - Compute pairwise cosine similarities
   - Divergence = 1 - mean(similarity)
   - Range: [0, 1], higher = more divergent

2. **Token Overlap** (secondary metric)
   - Compute ROUGE-L between response pairs
   - Divergence = 1 - ROUGE-L
   - Validates semantic similarity

3. **Answer Extraction + Consistency** (for QA tasks)
   - Extract key answer spans from each response
   - Check if all policies produce same answer
   - Divergence = fraction of disagreeing pairs

**Ensemble**: Combine metrics using weighted average (tune on validation set).

#### Step 3: Prepare Evaluation Data (Week 1, Day 2)
**Rationale**: Need annotated data where we know ground truth about memory gaps.

**Primary Dataset: LoCoMo**
- 10 conversations, ~300 turns each, 35 sessions
- Contains QA pairs where answers depend on conversation history
- Perfect for testing memory gap detection

**Data Processing**:
1. Load LoCoMo QA pairs
2. For each question, identify how many turns back the answer was mentioned
3. Create labels:
   - `recent`: answer in last 10 turns (should have low divergence)
   - `distant`: answer 10-50 turns back (should have medium divergence)
   - `very_distant`: answer 50+ turns back (should have high divergence)

**Secondary Dataset: SelfAware**
- 1,032 unanswerable + 2,337 answerable questions
- Tests general self-knowledge (not memory-specific)
- Use to validate that method generalizes

**Data Processing**:
1. Load SelfAware questions
2. No conversation history needed
3. Labels:
   - `answerable`: should have low divergence (all policies give same answer)
   - `unanswerable`: should have high divergence (policies differ on how to handle)

#### Step 4: Run Baseline Experiments (Week 1, Day 2-3)
**Rationale**: Need baselines to compare against for scientific rigor.

**Baseline 1: Always Answer with Best Policy**
- Use Policy 3 (RAG) to answer all questions
- No gap detection, no corrective action
- Metrics: Accuracy, F1

**Baseline 2: Verbalized Confidence**
- Ask model "How confident are you in this answer? (0-100)"
- Threshold on confidence to decide whether to answer
- Metrics: Accuracy, Coverage, Calibration (AUC)

**Baseline 3: Token Probability**
- Use mean log-probability of generated tokens as confidence
- Only applicable to open models or if we have access to logprobs
- Metrics: Accuracy, Coverage, Calibration (AUC)

**Baseline 4: Temperature Sampling Consistency**
- Generate 5 responses with temperature=0.7
- Measure self-consistency (semantic similarity)
- Use as uncertainty proxy
- Metrics: Accuracy, Coverage, Calibration

#### Step 5: Run Memory Gap Detection Experiments (Week 1, Day 3-4)
**Rationale**: Test our core hypothesis about divergence-based gap detection.

**Experiment 5a: Divergence vs. Distance**
- Research question: Does divergence increase with distance to relevant information?
- Method:
  1. For each LoCoMo QA pair, measure divergence score
  2. Correlate divergence with distance to answer (in turns)
  3. Plot divergence vs. distance
- Expected: Positive correlation (r > 0.5)
- Metrics: Spearman correlation, scatter plot

**Experiment 5b: Divergence vs. Correctness**
- Research question: Does divergence predict when the agent will answer incorrectly?
- Method:
  1. Run Policy 3 (RAG) on all LoCoMo questions
  2. Measure divergence score for each question
  3. Compute accuracy for each question
  4. Measure correlation and calibration
- Expected: High divergence → low accuracy
- Metrics: AUC (divergence as predictor of errors), calibration plot

**Experiment 5c: Corrective Actions**
- Research question: Do corrective actions improve performance?
- Method:
  1. When divergence > threshold, trigger action:
     - **Search**: Expand retrieval to top-20 instead of top-5
     - **Abstain**: Don't answer, report uncertainty
  2. Measure accuracy and coverage
  3. Compare against always-answer baseline
- Expected: Higher accuracy when abstaining on high-divergence cases
- Metrics: Precision-Coverage curve, Accuracy @ k coverage levels

#### Step 6: Validate on SelfAware (Week 1, Day 4)
**Rationale**: Test generalization to non-memory tasks.

**Experiment 6: Answerable vs. Unanswerable**
- Research question: Does divergence distinguish answerable from unanswerable?
- Method:
  1. Run policy simulation on SelfAware questions (no memory, so policies differ in approach)
  2. Measure divergence for answerable vs. unanswerable
  3. Classify questions using divergence threshold
- Expected: Unanswerable questions have higher divergence
- Metrics: Classification accuracy, F1, AUC

#### Step 7: Ablation Studies (Week 1, Day 5)
**Rationale**: Understand which components matter.

**Ablations**:
1. **Number of policies**: Test with 2, 3, 4, 5 policies
2. **Policy selection**: Which policies are most informative?
3. **Divergence metric**: Semantic vs. token overlap vs. answer consistency
4. **Threshold tuning**: How sensitive is performance to threshold?

### Baselines

**Memory Management Baselines**:
1. Full context (when possible)
2. RAG retrieval (standard approach)
3. Recency-based sliding window

**Uncertainty Estimation Baselines**:
1. Verbalized confidence ("How confident are you?")
2. Token probability (mean log-prob)
3. Temperature sampling consistency

**Task Baselines** (for LoCoMo):
1. Zero-shot (no memory)
2. Few-shot (with examples)
3. RAG baseline (from LoCoMo paper: ~58% accuracy)

### Evaluation Metrics

**Primary Metrics**:

1. **Gap Detection Accuracy**
   - Precision: When we detect a gap, is there actually one?
   - Recall: Of all actual gaps, how many do we detect?
   - F1 Score: Harmonic mean
   - Definition of "actual gap": Question is answered incorrectly due to missing memory

2. **Calibration (AUC)**
   - Treat divergence as confidence score
   - Plot calibration curve: predicted uncertainty vs. actual error rate
   - Compute Area Under Curve
   - Higher = better calibrated

3. **Task Performance with Actions**
   - Accuracy: Fraction of questions answered correctly
   - Coverage: Fraction of questions we choose to answer
   - Precision-Coverage curve: Trade-off between accuracy and coverage
   - Compare to baseline without gap detection

**Secondary Metrics**:

4. **Correlation Metrics**
   - Spearman correlation: divergence vs. distance to answer
   - Pearson correlation: divergence vs. error probability
   - Expected: Strong positive correlation

5. **Response Quality**
   - For questions we answer, measure:
     - Exact match accuracy
     - F1 (token overlap with ground truth)
     - ROUGE-L (for generated answers)

6. **Efficiency**
   - Computational cost: Time to run policy simulation
   - Cost: Number of LLM calls (4x for 4 policies)
   - Trade-off: Performance gain vs. computational overhead

**Statistical Tests**:
- Paired t-test: Our method vs. baselines on same test set
- Bootstrap resampling: 95% confidence intervals for all metrics
- Bonferroni correction: For multiple comparisons across experiments
- Significance level: p < 0.05

## Expected Outcomes

### Outcomes Supporting Hypothesis

**If H1 is correct**:
- Divergence score will have AUC > 0.75 for predicting answer errors
- Spearman correlation between divergence and distance > 0.5
- High-divergence cases will have <50% accuracy, low-divergence cases >80%

**If H2 is correct**:
- Abstaining on high-divergence cases will improve precision by >15%
- Searching memory on high-divergence cases will improve accuracy by >10%
- F1 score with actions will exceed F1 without actions by >5 points

**If H3 is correct**:
- Method will achieve >70% F1 on SelfAware answerable/unanswerable classification
- But will perform better on LoCoMo (memory-specific) than SelfAware (general)

### Outcomes Refuting Hypothesis

**If hypothesis is wrong**:
- Divergence will be uncorrelated with actual memory gaps (r < 0.2)
- Divergence won't predict errors better than random (AUC ~ 0.5)
- Corrective actions won't improve performance
- Method will fail to generalize to non-memory tasks

**Alternative explanations to consider**:
- Divergence might just reflect question difficulty, not memory gaps
- Different policies might have different capabilities independent of memory
- Computational cost might not justify small performance gains

## Resource Requirements

### Computational Resources
- **LLM API calls**: Estimated 4 policies × 100 LoCoMo questions × 5 runs = 2,000 calls
- **Cost estimate**: $50-100 for GPT-4 or Claude API calls
- **Time estimate**: ~2-3 hours for full LoCoMo evaluation
- **Hardware**: CPU sufficient (API-based), no GPU needed

### Datasets
- ✓ LoCoMo: Available in `datasets/locomo/`
- ✓ SelfAware: Available in `datasets/SelfAware/`
- Both datasets are already downloaded and validated

### Code Infrastructure
- LangChain memory-agent: Available in `code/memory-agent/`
- Uncertainty estimation methods: Available in `code/LLM-Uncertainty-Bench/`
- Will implement custom policy simulation and divergence measurement
- Language: Python 3.10+
- Key libraries: transformers, sentence-transformers, openai/anthropic, langchain

### Model Selection
**Primary Model**: Claude Sonnet 4.5 or GPT-4.1
- **Rationale**: State-of-the-art performance, good calibration, accessible API
- **Alternative**: GPT-4o-mini for development/testing (cheaper)

**Embedding Model**: all-MiniLM-L6-v2
- **Rationale**: Fast, lightweight, good for semantic similarity
- **Alternative**: text-embedding-3-small (OpenAI) if better performance needed

## Timeline and Milestones

### Phase 1: Setup and Implementation (3 hours)
- Hour 1: Environment setup, dependency installation
- Hour 2: Implement memory policies and divergence measurement
- Hour 3: Prepare LoCoMo and SelfAware datasets

**Deliverable**: Working policy simulation framework with unit tests

### Phase 2: Baseline Experiments (2 hours)
- Hour 4: Run baseline methods (verbalized confidence, sampling consistency)
- Hour 5: Evaluate baselines, document performance

**Deliverable**: Baseline results (accuracy, calibration, coverage)

### Phase 3: Core Experiments (3 hours)
- Hour 6: Run divergence vs. distance experiment (LoCoMo)
- Hour 7: Run divergence vs. correctness experiment
- Hour 8: Run corrective actions experiment

**Deliverable**: Core experimental results with statistical tests

### Phase 4: Validation and Ablations (2 hours)
- Hour 9: Run SelfAware validation
- Hour 10: Run ablation studies

**Deliverable**: Validation results and ablation analysis

### Phase 5: Analysis and Documentation (2 hours)
- Hour 11: Create visualizations, compute final metrics
- Hour 12: Write REPORT.md with comprehensive results

**Deliverable**: Complete research report with figures and tables

**Total Estimated Time**: 12 hours
**Buffer for Debugging**: +4 hours (33%)
**Total with Buffer**: 16 hours

## Potential Challenges and Mitigation

### Challenge 1: API Rate Limits and Cost
**Issue**: Running 4 policies per question with multiple models could hit rate limits or exceed budget
**Mitigation**:
- Use caching to avoid redundant API calls
- Implement exponential backoff for rate limit errors
- Start with small subset (20 questions) to estimate costs
- Use cheaper model (GPT-4o-mini) for development, upgrade to GPT-4.1 for final evaluation
**Fallback**: Reduce number of policies from 4 to 3, or subsample LoCoMo to 50 questions

### Challenge 2: Divergence Might Not Correlate with Memory Gaps
**Issue**: Core hypothesis might be wrong
**Mitigation**:
- Pilot on 10 questions first to check if signal exists
- If divergence is too noisy, try alternative metrics (answer consistency, token probability variance)
- Ablate to find which policies contribute most to useful divergence
**Fallback**: Reframe as "divergence indicates any uncertainty" rather than specifically memory gaps

### Challenge 3: LoCoMo Questions May Not Have Clear Answers in History
**Issue**: Dataset might be ambiguous or answers might not be in history
**Mitigation**:
- Manually inspect 20 LoCoMo QA pairs to validate quality
- Filter to questions with clear ground truth
- Create synthetic memory gap scenarios if needed
**Fallback**: Use SelfAware as primary dataset instead

### Challenge 4: Policies Might Be Too Similar
**Issue**: If policies produce nearly identical responses, divergence won't be informative
**Mitigation**:
- Use maximally different policies (full vs. none, recent vs. semantic)
- Add explicit instructions to differentiate policy behavior
- Test each policy independently to verify they behave differently
**Fallback**: Add more extreme policies (e.g., "intentionally forget" vs. "hallucinate from context")

### Challenge 5: Threshold Tuning Might Be Dataset-Specific
**Issue**: Optimal threshold for "high divergence" might not generalize
**Mitigation**:
- Use validation set to tune thresholds
- Test on held-out test set
- Report performance across multiple thresholds (precision-coverage curve)
**Fallback**: Use adaptive thresholding based on percentiles

### Challenge 6: Computational Cost Too High
**Issue**: 4 policies × N questions might take too long or cost too much
**Mitigation**:
- Batch API requests to improve throughput
- Cache responses aggressively
- Parallelize where possible
**Fallback**: Reduce to 2-3 policies, or subsample dataset

## Success Criteria

### Minimum Viable Success
- **H1**: Divergence predicts errors with AUC > 0.65 (better than random, demonstrates signal exists)
- **H2**: Corrective actions improve F1 by at least 5 points over naive baseline
- **H3**: Method achieves >65% F1 on SelfAware (competitive with simple baselines)
- **Reproducibility**: All experiments run successfully with documented results

### Target Success
- **H1**: Divergence predicts errors with AUC > 0.75 (strong predictive power)
- **H2**: Corrective actions improve F1 by 10+ points, precision by 15+ points
- **H3**: Method achieves >75% F1 on SelfAware, better on LoCoMo
- **Statistical significance**: p < 0.05 on all primary comparisons
- **Interpretability**: Clear visualizations showing when/why divergence indicates gaps

### Exceptional Success
- **H1**: AUC > 0.85, rivaling supervised methods
- **H2**: F1 improvement > 15 points with minimal coverage loss
- **H3**: Generalizes across both datasets and to out-of-distribution examples
- **Novel insights**: Discover which types of memory gaps are easiest/hardest to detect
- **Practical impact**: Demonstrate actionable framework for real-world deployment

### Research Contribution Regardless of Results

Even if hypothesis is partially refuted:
- **Negative result is valuable**: Knowing that divergence doesn't work rules out this approach
- **Methodology is novel**: Policy simulation framework is reusable
- **Benchmarking is useful**: Comprehensive comparison of uncertainty methods on memory tasks
- **Dataset analysis**: Characterization of LoCoMo in terms of memory demands

## Ethical Considerations

### Potential Harms
- Memory gap detection could be used to hide information from users ("I don't remember" when inconvenient)
- Could exacerbate issues if used to avoid accountability in sensitive domains (medical, legal)

### Mitigation
- Focus on user-beneficial applications (e.g., preventing harmful recommendations)
- Document limitations and potential for misuse
- Emphasize transparency: agents should be honest about uncertainty

### Positive Impact
- Safer dialogue agents that avoid confidently wrong answers
- Better user experience through appropriate help-seeking behavior
- Foundation for agents that can explain their own limitations

## Documentation Plan

### Code Documentation
- Docstrings for all functions and classes
- README with setup instructions and example usage
- Comments explaining key algorithmic decisions
- Unit tests for policy simulation and divergence measurement

### Experimental Documentation
- Config files for all experiments (hyperparameters, model versions)
- Logging of all API calls and responses
- Version tracking for datasets and models
- Random seeds for reproducibility

### Results Documentation
- REPORT.md with comprehensive findings (following template)
- Figures with clear captions
- Tables with statistical tests
- Discussion of limitations and future work

## Summary

This research plan provides a systematic approach to testing whether dialogue agents can detect their own memory gaps through policy simulation and divergence measurement. The approach is:

✓ **Grounded in literature**: Builds on recent work in memory systems, self-knowledge, and uncertainty estimation
✓ **Scientifically rigorous**: Includes multiple baselines, statistical tests, ablations
✓ **Feasible**: Uses available datasets, reasonable compute requirements
✓ **Novel**: First work to combine policy simulation with memory gap detection
✓ **Actionable**: Provides framework for real-world deployment

**Key Innovations**:
1. Memory policy simulation as meta-cognitive framework
2. Response divergence as memory gap signal
3. Integration with corrective actions

**Expected Contributions**:
1. Novel method for memory gap detection
2. Comprehensive evaluation on long-term dialogue (LoCoMo)
3. Validation on general self-knowledge (SelfAware)
4. Actionable framework for safer dialogue agents

The plan balances ambition (testing novel hypothesis) with pragmatism (using available resources, reasonable timeline). Even if core hypothesis is refuted, the systematic evaluation will advance understanding of memory and uncertainty in dialogue agents.
