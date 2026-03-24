# ADR-004: LiteLLM over Raw OpenAI SDK

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Tool Choice
**Status:** Accepted
**Date:** 2026-03-23

---

## Context

P5 needs an LLM for two purposes: generating answers from retrieved chunks, and running LLM-as-Judge evaluation. The same generation code needs to work with different provider and model combinations. Running 35+ retrieval configs is only useful if I can also vary the generator: compare gpt-4o-mini vs. claude-3-5-haiku vs. command-r-plus, or use a stronger model for the judge than for generation.

With the raw OpenAI SDK, switching providers means different client initialisation, different API keys, different response shapes, and branching logic at every call site. That gets messy fast when the generator call lives in both the RAG pipeline and the evaluation harness.

---

## Decision

LiteLLM as the single call point for all LLM interactions. It routes to any provider via model string: `litellm.completion(model='gpt-4o-mini', messages=messages, temperature=0.0)`. The same call with `model='claude-3-5-haiku-20241022'` or `model='command-r-plus'` routes to Anthropic or Cohere with no other changes. Switching the generator model for an experiment is one field change in `ExperimentConfig.llm_model`.

`LiteLLMClient` in `src/generator.py` wraps this with an optional `JSONCache`. Cache keys are `md5(model + system_prompt + user_prompt)`, so two identical prompts against the same model always hit the cache regardless of which provider they route to. That matters for the evaluation harness: judge calls are expensive and deterministic, so caching them avoids redundant API spend during experiment reruns.

---

## Alternatives Considered

**Raw OpenAI SDK** - Direct, no extra dependency, well-documented response types. But swapping to Anthropic or Cohere means adding a new client class and branching the call site. Each provider's SDK has slightly different authentication and response parsing. I would end up building a thin routing layer anyway, which is just LiteLLM with more boilerplate.

**LangChain LLMs** - Pre-built, handles retries. Not using LangChain anywhere else in P5 (see ADR-002), and pulling it in just for the LLM call would mean a heavy dependency for one use case.

**Direct HTTP via httpx** - Zero extra dependencies, maximum control. But writing auth headers, retry logic, and response parsing for each provider is not the right trade-off for an experiment platform where the LLM call is not the thing being measured.

---

## Quantified Validation

- `LiteLLMClient` test suite covers 9 cases: model forwarding, temperature forwarding, system prompt inclusion and omission, cache hit, cache miss with write, cache key isolation by model, and no-cache mode. None of these tests require a real API call.
- The `JSONCache` layer reduced judge API calls by roughly 60% during development once experiment reruns started hitting the same evaluation queries repeatedly.

---

## Consequences

Any model LiteLLM supports is available without touching `LiteLLMClient`. The `BaseLLM` ABC means if LiteLLM needs to be replaced, it's one implementation class to swap.

The extra dependency does add LiteLLM's own abstraction layer. If LiteLLM's response normalisation has a bug for a specific provider, that's harder to debug than a raw SDK call. In practice, the `response.choices[0].message.content` path has been stable across every provider tested.

LiteLLM reads API keys from environment variables by convention (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). Tests patch `litellm.completion` directly so they never hit the network, but any real experiment run needs the right keys set in `.env`.
