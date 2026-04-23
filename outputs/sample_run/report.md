# HotpotQA-100 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: openai
- Questions: 100
- Total records (all agents): 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.56 | 0.8 | 0.24 |
| Avg attempts | 1 | 1.74 | 0.74 |
| Avg tokens per question | 3261.05 | 8191.24 | 4930.19 |
| Avg latency (ms) | 3904.41 | 11972.16 | 8067.75 |

## Failure modes
```json
{
  "react": {
    "none": 56,
    "wrong_final_answer": 44
  },
  "reflexion": {
    "none": 80,
    "wrong_final_answer": 20
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts

## Discussion
This benchmark was executed on HotpotQA with 100 questions using real LLM calls. Reflexion usually improves exact match on multi-hop cases by using evaluator feedback and strategy memory, but the gain comes with extra attempts, tokens, and latency. The most useful follow-up analysis is to inspect errors by failure mode, then check whether low-quality reflection text or strict evaluator behavior is the bottleneck.
