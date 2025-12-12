"""
Memory Gap Detection System using Policy Simulation

This module implements a meta-cognitive framework for detecting memory gaps in dialogue agents
by simulating responses under multiple memory policies and measuring response divergence.
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from tqdm import tqdm


@dataclass
class MemoryPolicy:
    """Represents a memory policy for conversation context."""
    name: str
    description: str

    def apply(self, conversation_history: List[Dict[str, str]], current_query: str) -> str:
        """Apply this policy to select relevant conversation context."""
        raise NotImplementedError


class FullContextPolicy(MemoryPolicy):
    """Include all conversation history."""

    def __init__(self):
        super().__init__(
            name="full_context",
            description="Include all conversation history"
        )

    def apply(self, conversation_history: List[Dict[str, str]], current_query: str) -> str:
        """Return all conversation history as context."""
        context_parts = []
        for i, turn in enumerate(conversation_history):
            speaker = turn.get('speaker', turn.get('role', 'unknown'))
            text = turn.get('text', turn.get('content', ''))
            context_parts.append(f"Turn {i+1} [{speaker}]: {text}")
        return "\n".join(context_parts)


class RecentOnlyPolicy(MemoryPolicy):
    """Include only recent conversation turns."""

    def __init__(self, k: int = 10):
        super().__init__(
            name=f"recent_{k}",
            description=f"Include only last {k} conversation turns"
        )
        self.k = k

    def apply(self, conversation_history: List[Dict[str, str]], current_query: str) -> str:
        """Return only recent K turns as context."""
        recent_history = conversation_history[-self.k:]
        context_parts = []
        for i, turn in enumerate(recent_history):
            speaker = turn.get('speaker', turn.get('role', 'unknown'))
            text = turn.get('text', turn.get('content', ''))
            context_parts.append(f"Turn {len(conversation_history)-len(recent_history)+i+1} [{speaker}]: {text}")
        return "\n".join(context_parts)


class SemanticRetrievalPolicy(MemoryPolicy):
    """Retrieve most relevant turns using semantic similarity."""

    def __init__(self, k: int = 5, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(
            name=f"semantic_{k}",
            description=f"Retrieve top-{k} semantically relevant turns"
        )
        self.k = k
        self.model = SentenceTransformer(model_name)

    def apply(self, conversation_history: List[Dict[str, str]], current_query: str) -> str:
        """Retrieve and return semantically similar turns."""
        if len(conversation_history) == 0:
            return ""

        # Extract text from history
        history_texts = []
        for turn in conversation_history:
            text = turn.get('text', turn.get('content', ''))
            history_texts.append(text)

        # Encode query and history
        query_emb = self.model.encode([current_query], convert_to_tensor=False)
        history_embs = self.model.encode(history_texts, convert_to_tensor=False)

        # Compute similarities
        similarities = cosine_similarity(query_emb, history_embs)[0]

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]

        # Build context from top-k turns
        context_parts = []
        for idx in top_k_indices:
            turn = conversation_history[idx]
            speaker = turn.get('speaker', turn.get('role', 'unknown'))
            text = turn.get('text', turn.get('content', ''))
            context_parts.append(f"Turn {idx+1} [{speaker}]: {text}")

        return "\n".join(context_parts)


class NoMemoryPolicy(MemoryPolicy):
    """No conversation history, current turn only."""

    def __init__(self):
        super().__init__(
            name="no_memory",
            description="No conversation history, current turn only"
        )

    def apply(self, conversation_history: List[Dict[str, str]], current_query: str) -> str:
        """Return empty context (no memory)."""
        return ""


class ResponseGenerator:
    """Generates responses using LLM APIs."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize response generator with specified model."""
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, context: str, query: str, temperature: float = 0.7) -> str:
        """Generate response given context and query."""
        # Build prompt
        if context:
            system_message = "You are a helpful dialogue agent. Use the conversation history to answer the user's question."
            user_message = f"Conversation History:\n{context}\n\nCurrent Question: {query}\n\nAnswer:"
        else:
            system_message = "You are a helpful dialogue agent."
            user_message = f"Question: {query}\n\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[Error: {str(e)}]"


class DivergenceMeasurer:
    """Measures divergence between responses."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence embedding model."""
        self.model = SentenceTransformer(model_name)

    def compute_semantic_similarity(self, responses: List[str]) -> float:
        """Compute pairwise semantic similarity between responses."""
        if len(responses) < 2:
            return 1.0

        # Encode responses
        embeddings = self.model.encode(responses, convert_to_tensor=False)

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embeddings)

        # Get upper triangle (avoid diagonal and duplicates)
        n = len(responses)
        pairwise_sims = []
        for i in range(n):
            for j in range(i+1, n):
                pairwise_sims.append(similarities[i][j])

        # Return mean similarity as Python float
        return float(np.mean(pairwise_sims)) if pairwise_sims else 1.0

    def compute_divergence(self, responses: List[str]) -> float:
        """Compute divergence score (1 - similarity)."""
        similarity = self.compute_semantic_similarity(responses)
        divergence = 1.0 - similarity
        # Convert to Python float and clamp to [0, 1]
        return float(max(0.0, min(1.0, divergence)))


class MemoryGapDetector:
    """Main system for detecting memory gaps through policy simulation."""

    def __init__(
        self,
        policies: Optional[List[MemoryPolicy]] = None,
        response_generator: Optional[ResponseGenerator] = None,
        divergence_measurer: Optional[DivergenceMeasurer] = None
    ):
        """Initialize gap detector with policies and components."""
        # Default policies if none provided
        if policies is None:
            self.policies = [
                FullContextPolicy(),
                RecentOnlyPolicy(k=10),
                SemanticRetrievalPolicy(k=5),
                NoMemoryPolicy()
            ]
        else:
            self.policies = policies

        # Components
        self.response_generator = response_generator or ResponseGenerator()
        self.divergence_measurer = divergence_measurer or DivergenceMeasurer()

    def detect_gap(
        self,
        conversation_history: List[Dict[str, str]],
        query: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect memory gap for a query given conversation history.

        Returns:
            Dictionary with:
            - responses: Dict[policy_name, response]
            - divergence_score: float in [0, 1]
            - gap_detected: bool
            - recommended_action: str
        """
        # Simulate responses under each policy
        responses = {}
        for policy in self.policies:
            # Apply policy to get context
            context = policy.apply(conversation_history, query)

            # Generate response with this context
            response = self.response_generator.generate(context, query, temperature)
            responses[policy.name] = response

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        # Measure divergence
        response_texts = list(responses.values())
        divergence_score = self.divergence_measurer.compute_divergence(response_texts)

        # Determine gap detection and action
        gap_detected, recommended_action = self._determine_action(divergence_score)

        return {
            'responses': responses,
            'divergence_score': divergence_score,
            'gap_detected': gap_detected,
            'recommended_action': recommended_action
        }

    def _determine_action(self, divergence_score: float) -> Tuple[bool, str]:
        """Determine if gap exists and recommended action based on divergence."""
        # Thresholds (can be tuned on validation set)
        HIGH_THRESHOLD = 0.4
        MEDIUM_THRESHOLD = 0.2

        if divergence_score > HIGH_THRESHOLD:
            return True, "SEARCH_MEMORY"
        elif divergence_score > MEDIUM_THRESHOLD:
            return True, "EXPRESS_UNCERTAINTY"
        else:
            return False, "ANSWER_CONFIDENTLY"


def load_locomo_data(file_path: str) -> List[Dict]:
    """Load LoCoMo dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_selfaware_data(file_path: str) -> List[Dict]:
    """Load SelfAware dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    # Note: OpenAI API doesn't support seed parameter in all models
    # We use temperature to control stochasticity instead


if __name__ == "__main__":
    # Example usage
    set_seed(42)

    # Initialize detector
    detector = MemoryGapDetector()

    # Example conversation
    conversation = [
        {"speaker": "user", "text": "My name is Alice and I'm allergic to peanuts."},
        {"speaker": "agent", "text": "Nice to meet you, Alice. I'll remember that you're allergic to peanuts."},
        {"speaker": "user", "text": "What's the weather like today?"},
        {"speaker": "agent", "text": "I don't have access to current weather data, but I can help you find that information."}
    ]

    # Test query
    query = "Can you recommend a dessert for me?"

    # Detect gap
    result = detector.detect_gap(conversation, query)

    print(f"Query: {query}")
    print(f"\nDivergence Score: {result['divergence_score']:.3f}")
    print(f"Gap Detected: {result['gap_detected']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print("\nResponses by policy:")
    for policy_name, response in result['responses'].items():
        print(f"\n{policy_name}: {response}")
