# Migration Guide

Step-by-step instructions for migrating to `llm-std-lib` from common alternatives.

---

## From direct OpenAI SDK calls

**Before:**

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
text = response.choices[0].message.content
```

**After:**

```python
from llm_std_lib import LLMClient, LLMConfig

client = LLMClient(config=LLMConfig.from_env())
response = await client.acomplete(prompt="Hello", model="openai/gpt-4o-mini")
text = response.text
```

**What you gain:** caching, routing, resilience, metrics — all optional, zero config to start.

---

## From direct Anthropic SDK calls

**Before:**

```python
import anthropic

client = anthropic.AsyncAnthropic(api_key="sk-ant-...")
message = await client.messages.create(
    model="claude-haiku-20240307",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
text = message.content[0].text
```

**After:**

```python
config = LLMConfig(providers={"anthropic": {"api_key": "sk-ant-..."}})
client = LLMClient(config=config)
response = await client.acomplete(prompt="Hello", model="anthropic/claude-haiku-3")
text = response.text
```

---

## From LiteLLM

LiteLLM and llm-std-lib share the `provider/model` string convention — migration is mostly
a find-and-replace.

**Before:**

```python
import litellm

response = await litellm.acompletion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
text = response.choices[0].message.content
```

**After:**

```python
from llm_std_lib import LLMClient, LLMConfig

client = LLMClient(config=LLMConfig.from_env())
response = await client.acomplete(prompt="Hello", model="openai/gpt-4o-mini")
text = response.text
```

**Key differences:**

| Aspect | LiteLLM | llm-std-lib |
| --- | --- | --- |
| Semantic cache | via `litellm.cache` | built-in `SemanticCache` |
| Routing | manual | `ModelRouter` strategies |
| Circuit breaker | not built-in | `ResilienceEngine` |
| Metrics | not built-in | Prometheus / OTLP |
| Config from Vault | not built-in | `VaultConfig` |

---

## From LangChain

LangChain's LLM wrappers can be replaced with `LLMClient` for simpler use cases.

**Before:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")
result = await llm.ainvoke([HumanMessage(content="Hello")])
text = result.content
```

**After:**

```python
client = LLMClient(config=LLMConfig.from_env())
response = await client.acomplete(prompt="Hello", model="openai/gpt-4o-mini")
text = response.text
```

**When to keep LangChain:** llm-std-lib does not replace LangChain's agent framework,
tool-calling abstractions, or document loaders. Use llm-std-lib as the transport layer
*inside* LangChain chains by implementing a custom `BaseChatModel` wrapper if needed.

---

## Response field mapping

| Old SDK field | llm-std-lib field |
| --- | --- |
| `choices[0].message.content` | `response.text` |
| `usage.prompt_tokens` | `response.prompt_tokens` |
| `usage.completion_tokens` | `response.completion_tokens` |
| `usage.total_tokens` | `response.total_tokens` |
| *(calculated manually)* | `response.cost_usd` |
| *(measured manually)* | `response.latency_ms` |
| *(not available)* | `response.from_cache` |
