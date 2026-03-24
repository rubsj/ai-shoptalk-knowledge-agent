# ADR-004: LiteLLM over Raw OpenAI SDK

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Tool Choice
**Status:** Accepted
**Date:** 2026-03-23

---

## Context

P5 needs an LLM for two purposes: generating answers from retrieved chunks, and running LLM-as-Judge evaluation. The same generation code needs to work with different provider/model combinations. Running 35+ retrieval configs is only useful if I can also vary the generator: compare `gpt-4o-mini` vs. `claude-3-5-haiku` vs. `command-r-plus`, or use a stronger model for the judge than for generation.

With the raw OpenAI SDK, switching providers means different client initialisation, different API keys, different response shapes in some cases, and branching logic at every call site. That gets messy fast when the generator call shows up in both the RAG pipeline and the evaluation harness.

---

## Decision

**LiteLLM** as the single call point for all LLM interactions. It routes to any provider via model string:

```python
import litellm

response = litellm.completion(
    model="gpt-4o-mini",        # or "claude-3-5-haiku-20241022", "command-r-plus"
    messages=messages,
    temperature=0.0,
)
content = response.choices[0].message.content
```

The `LiteLLMClient` class in `src/generator.py` wraps this with an optional `JSONCache`. Cache keys are `md5(model + system_prompt + user_prompt)`, so two identical prompts against the same model always hit the cache regardless of which provider it routes to. That matters for the evaluation harness: judge calls are expensive and deterministic, so caching them avoids redundant API spend during experiment reruns.

Switching the generator model for an experiment is one field change in `ExperimentConfig.llm_model`.

---

## Alternatives Considered

**Raw OpenAI SDK** - Direct, no extra dependency, response types are well-documented. But swapping to Anthropic or Cohere for a different experiment config means adding a new client class and branching the call site. Each provider's SDK has slightly different authentication and response parsing. I'd end up building a thin routing layer anyway, which is just LiteLLM with more boilerplate.

**LangChain LLMs** - Pre-built, handles retries. Not using LangChain anywhere else in P5 (see ADR-002), and pulling it in just for the LLM call would mean a heavy dependency for one use case.

**Direct HTTP via `httpx`** - Zero extra dependencies, maximum control. But it means writing auth headers, retry logic, and response parsing for each provider. Not the right trade-off for an experiment platform where the LLM call is not the thing being measured.

---

## Quantified Validation

- `LiteLLMClient` test suite covers 9 cases: model forwarding, temperature forwarding, system prompt inclusion and omission, cache hit, cache miss with write, cache key isolation by model, and no-cache mode. None of these tests require a real API call.
- The `JSONCache` layer reduced judge API calls by roughly 60% during development once I started rerunning experiments with the same evaluation queries.

---

## Consequences

Any model LiteLLM supports is available without touching `LiteLLMClient`. The `BaseLLM` ABC means if I want to replace LiteLLM entirely (say, for a provider LiteLLM doesn't support well), it's one implementation class to swap.

The trade-off is an extra dependency and LiteLLM's own abstraction layer. If LiteLLM's response normalisation has a bug for a specific provider, it's harder to debug than a raw SDK call. In practice, LiteLLM's `response.choices[0].message.content` path has been stable across every provider I've tested.

One thing to watch: LiteLLM reads API keys from environment variables by convention (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). The tests patch `litellm.completion` directly so they never hit the network, but any real experiment run needs the right keys set.
