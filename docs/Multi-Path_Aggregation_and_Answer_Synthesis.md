# Multi-Path Aggregation and Answer Synthesis

This document explains how AGC-Agent aggregates multiple reasoning paths and synthesizes final answers for Knowledge Graph Question Answering (KGQA).

## Overview

After the Agentic Reasoning Controller completes beam search exploration, the system produces multiple candidate reasoning paths. The **Multi-Path Aggregation** module processes these paths, and the **Answer Synthesis** module produces the final structured output.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGC-Agent Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────┐     ┌────────────────────────┐    │
│  │   Beam      │     │   Path          │     │   Answer               │    │
│  │   Search    │ ──► │   Accumulator   │ ──► │   Synthesis            │    │
│  │             │     │                 │     │                        │    │
│  │ (Multiple   │     │ (Collects &     │     │ (Produces final        │    │
│  │  paths)     │     │  ranks paths)   │     │  structured output)    │    │
│  └─────────────┘     └────────┬────────┘     └────────────────────────┘    │
│                               │                                             │
│                               ▼                                             │
│                      ┌─────────────────┐                                    │
│                      │   Multi-Path    │                                    │
│                      │   Aggregation   │                                    │
│                      │                 │                                    │
│                      │ (LLM extracts   │                                    │
│                      │  answers from   │                                    │
│                      │  each path)     │                                    │
│                      └─────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Path Accumulator

The `PathAccumulator` class (`agc_agent/beam_state.py:452-687`) serves as the memory system that collects and manages reasoning paths from completed beams.

### 1.1 Data Structure

```python
class PathAccumulator:
    def __init__(self):
        self.paths: List[Tuple[str, float]] = []  # (path_string, score)
```

Each path is stored as:
- **path_string**: Formatted reasoning path (e.g., `"USA -> president -> Barack Obama"`)
- **score**: Cumulative confidence score from beam search

### 1.2 Path Collection

Paths are collected from `BeamState` objects after beam search completes:

```python
def add_path(self, beam: BeamState) -> None:
    """Add a completed beam's path."""
    path_str = beam.path_to_string()  # "e_0 -> r_1 -> e_1 -> r_2 -> e_2 -> ..."
    self.paths.append((path_str, beam.cumulative_score))

def add_paths(self, beams: List[BeamState]) -> None:
    """Add multiple beams' paths."""
    for beam in beams:
        self.add_path(beam)
```

### 1.3 Path Format

Paths follow the GCR-compatible format:

```
e_0 -> r_1 -> e_1 -> r_2 -> e_2 -> ... -> r_n -> e_n
```

**Example:**
```
USA -> government.country.presidents -> Barack Obama -> people.person.spouse -> Michelle Obama
```

---

## 2. Multi-Path Aggregation

The Multi-Path Aggregation module uses an LLM to extract answers from reasoning paths. This is more sophisticated than naive last-entity extraction, as it identifies which entity in each path actually answers the question.

### 2.1 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Path Aggregation Flow                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Input: Question + Candidate Paths                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Question: "Who is the spouse of the 44th president of the USA?"     │  │
│   │                                                                     │  │
│   │ Paths (ranked by confidence):                                       │  │
│   │   [1] (0.95) USA -> president -> Barack Obama -> spouse -> Michelle │  │
│   │   [2] (0.87) USA -> president -> Donald Trump -> spouse -> Melania  │  │
│   │   [3] (0.72) USA -> first_lady -> Michelle Obama                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              LLM Answer Extraction (per path)                       │  │
│   │                                                                     │  │
│   │   For each path:                                                    │  │
│   │     1. Extract all entities from path                               │  │
│   │     2. Build prompt with question + path + entity list              │  │
│   │     3. LLM identifies which entity answers the question             │  │
│   │     4. Validate output against entity list                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                            │
│                               ▼                                            │
│   Output: Answers with supporting paths                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   Answer: Michelle Obama                                            │  │
│   │   Path: USA -> president -> Barack Obama -> spouse -> Michelle      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 LLM Answer Extraction

The core extraction logic is in `PathAccumulator._extract_answer_with_llm()`:

```python
def _extract_answer_with_llm(
    self,
    question: str,
    path_str: str,
    model,
    tokenizer,
    topic_entities: List[str] = None
) -> str:
    # 1. Extract all entities from the path
    parts = path_str.split(" -> ")
    entities = [parts[i] for i in range(0, len(parts), 2)]  # entities at even indices

    # 2. Build prompt for answer extraction
    # 3. Generate with LLM
    # 4. Validate and return matched entity
```

### 2.3 Aggregation Prompt Template

From `prompt/agent_prompts.py`, the aggregation system uses:

**System Prompt:**
```
You are an expert knowledge graph reasoner performing GROUNDED answer extraction.

CRITICAL: You can ONLY output answers that appear in the provided reasoning paths.
You CANNOT invent, imagine, or suggest any entity that is not explicitly present in the paths.

Your task is to extract answers from the provided Candidate Reasoning Paths that correctly answer the question.

VERIFICATION PROCESS:
1. Read the Question and identify what TYPE of entity is being asked for
2. For EACH candidate path, examine:
   a) Which entity in the path matches the TYPE the question asks for?
   b) Is there a relation that semantically connects to what the question asks?
   c) Is the reasoning chain logically valid?
3. ONLY include answers where the path logically supports the answer.

Output format (one line per answer):
# Answer: [Entity Name], #Reasoning Path: [Full Path String]
```

**User Prompt:**
```
Question: {question}

Reasoning Path: {path_str}

Entities in path:
- {entity_1}
- {entity_2}
- ...

Which entity answers the question? Output only the entity name, nothing else.
```

### 2.4 Answer Validation

The extraction includes validation to prevent hallucination:

```python
# Try to match output to one of the non-topic entities
for entity in sorted(set(non_topic_entities), key=len, reverse=True):
    # Exact match first
    if first_line.lower().strip() == entity.lower():
        return entity
    # Then containment
    if entity.lower() in first_line.lower():
        return entity

# Fallback to first non-topic entity
if non_topic_entities:
    return non_topic_entities[0]
```

Key validation steps:
1. **Topic Entity Filtering**: Excludes starting entities (they're questions, not answers)
2. **Exact Match**: Prefers exact entity name matches
3. **Containment Match**: Falls back to substring matching
4. **Safe Fallback**: Returns first non-topic entity if no match found

---

## 3. Answer Synthesis

The Answer Synthesis module produces the final structured output that matches the GCR evaluation format.

### 3.1 Output Structure

From `agc_agent/agc_agent.py`:

```python
@dataclass
class AGCAgentResult:
    """Result from AGC-Agent reasoning."""
    question: str
    predictions: List[str]           # Formatted paths for evaluation
    answers: List[Tuple[str, float]] # (answer, confidence) pairs
    reasoning_trace: Dict[str, Any]  # Statistics and metadata
    raw_paths: List[Tuple[str, float]] # (path_string, score) pairs
```

### 3.2 Formatted Output for Evaluation

Each prediction follows the GCR-compatible format:

```
# Reasoning Path:
{path_string}
# Answer:
{extracted_answer}
```

**Example:**
```
# Reasoning Path:
USA -> government.country.presidents -> Barack Obama -> people.person.spouse -> Michelle Obama
# Answer:
Michelle Obama
```

### 3.3 Reasoning Trace

The reasoning trace captures search statistics:

```python
reasoning_trace = {
    "total_paths_explored": N,    # Total beams processed
    "completed_paths": K,          # Beams that found answers
    "max_depth_reached": D,        # Maximum reasoning depth
    "backtrack_count": B           # Total backtracks across all beams
}
```

### 3.4 Full Synthesis Flow

```python
def reason(self, question, graph_triples, topic_entities) -> AGCAgentResult:
    # 1. Initialize and run beam search
    result_beams = self._run_beam_search(question, topic_entities)

    # 2. Accumulate paths
    accumulator = PathAccumulator()
    accumulator.add_paths(result_beams)

    # 3. Format for evaluation using LLM-based answer extraction
    predictions = accumulator.format_for_evaluation_with_llm(
        question=question,
        model=self.model,
        tokenizer=self.tokenizer,
        top_k=self.config.output_top_k,
        topic_entities=topic_entities
    )

    # 4. Get unique answers with scores
    answers = accumulator.get_answers_with_llm(
        question=question,
        model=self.model,
        tokenizer=self.tokenizer,
        topic_entities=topic_entities
    )

    # 5. Compile statistics
    stats = self.beam_manager.get_statistics()

    # 6. Return structured result
    return AGCAgentResult(
        question=question,
        predictions=predictions,
        answers=answers,
        reasoning_trace={...},
        raw_paths=raw_paths
    )
```

---

## 4. Integration with Evaluation

### 4.1 Output File Format

Results are saved as JSONL (one JSON object per line):

```json
{
    "id": "sample_123",
    "question": "Who is the spouse of Barack Obama?",
    "prediction": [
        "# Reasoning Path:\nBarack Obama -> spouse -> Michelle Obama\n# Answer:\nMichelle Obama",
        "# Reasoning Path:\nBarack Obama -> family.member -> Michelle Obama\n# Answer:\nMichelle Obama"
    ],
    "ground_truth": ["Michelle Obama"],
    "ground_truth_paths": ["Barack Obama -> spouse -> Michelle Obama"],
    "reasoning_trace": {
        "total_paths_explored": 15,
        "completed_paths": 5,
        "max_depth_reached": 2,
        "backtrack_count": 2
    }
}
```

### 4.2 Evaluation Metrics

The output is compatible with GCR's evaluation functions:

- **Hit@K**: Whether any of the top-K predictions matches ground truth
- **F1 Score**: Precision-recall balance for multi-answer questions
- **Path Accuracy**: Whether reasoning paths are valid in the KG

---

## 5. Comparison: Naive vs LLM-Based Extraction

### 5.1 Naive Extraction (Last Entity)

```python
def format_for_evaluation(self, top_k: int = -1) -> List[str]:
    """Naive: always returns last entity as answer."""
    for path_str, score in paths:
        parts = path_str.split(" -> ")
        answer = parts[-1]  # Last entity
        formatted = f"# Reasoning Path:\n{path_str}\n# Answer:\n{answer}"
```

**Limitation**: Assumes the answer is always the last entity, which fails for:
- Questions about intermediate entities
- Paths that continue past the answer

### 5.2 LLM-Based Extraction

```python
def format_for_evaluation_with_llm(self, question, model, tokenizer, top_k, topic_entities):
    """LLM-based: identifies correct answer entity semantically."""
    for path_str, score in paths:
        answer = self._extract_answer_with_llm(question, path_str, model, tokenizer, topic_entities)
        formatted = f"# Reasoning Path:\n{path_str}\n# Answer:\n{answer}"
```

**Advantages**:
- Semantically identifies the answer based on question type
- Handles complex reasoning chains
- Filters out topic entities (prevents returning the question subject as answer)

---

## 6. Code Reference

| Component | File | Line |
|-----------|------|------|
| PathAccumulator | `agc_agent/beam_state.py` | 452-687 |
| AGCAgentResult | `agc_agent/agc_agent.py` | 61-68 |
| AGCAgent.reason() | `agc_agent/agc_agent.py` | 203-261 |
| AggregatorPromptBuilder | `prompt/agent_prompts.py` | 223-268 |
| AggregatorResponseParser | `prompt/agent_prompts.py` | 324-355 |
| process_sample() | `agc_reasoning.py` | 104-178 |

---

## 7. Example End-to-End Flow

### Input

```python
question = "Who is the spouse of the 44th president of the USA?"
topic_entities = ["USA"]
graph_triples = [
    ("USA", "government.country.presidents", "Barack Obama"),
    ("Barack Obama", "people.person.spouse", "Michelle Obama"),
    ...
]
```

### Beam Search Output

```python
result_beams = [
    BeamState(
        current_entity="Michelle Obama",
        path=[
            ("USA", "government.country.presidents", "Barack Obama"),
            ("Barack Obama", "people.person.spouse", "Michelle Obama")
        ],
        cumulative_score=0.95,
        status=COMPLETED
    ),
    ...
]
```

### Path Accumulation

```python
accumulator.paths = [
    ("USA -> government.country.presidents -> Barack Obama -> people.person.spouse -> Michelle Obama", 0.95),
    ...
]
```

### LLM Answer Extraction

**Prompt sent to LLM:**
```
Question: Who is the spouse of the 44th president of the USA?

Reasoning Path: USA -> government.country.presidents -> Barack Obama -> people.person.spouse -> Michelle Obama

Entities in path:
- USA
- Barack Obama
- Michelle Obama

Which entity answers the question? Output only the entity name, nothing else.
```

**LLM Response:** `Michelle Obama`

### Final Output

```json
{
    "question": "Who is the spouse of the 44th president of the USA?",
    "predictions": [
        "# Reasoning Path:\nUSA -> government.country.presidents -> Barack Obama -> people.person.spouse -> Michelle Obama\n# Answer:\nMichelle Obama"
    ],
    "answers": [("Michelle Obama", 0.95)],
    "reasoning_trace": {
        "total_paths_explored": 12,
        "completed_paths": 3,
        "max_depth_reached": 2,
        "backtrack_count": 1
    }
}
```

---

## 8. Configuration Options

From `AGCAgentConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_top_k` | 10 | Number of paths to include in output |
| `beam_width` | 10 | Maximum concurrent beams during search |
| `answer_threshold` | 0.5 | Minimum confidence to accept as answer |

These parameters affect how many paths are aggregated and synthesized into the final output.
