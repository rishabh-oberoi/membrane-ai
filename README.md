<div align="center">

# Membrane
**Zero-latency, reversible PII anonymization middleware for LLMs.**

[![PyPI Version](https://img.shields.io/pypi/v/membrane-ai.svg)](https://pypi.org/project/membrane-ai/)
[![Downloads](https://img.shields.io/pypi/dm/membrane-ai.svg)](https://pypi.org/project/membrane-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

> [!IMPORTANT]
> **Membrane Enterprise**  
> The open-source SDK is built for developers and rapid prototyping. For production deployments in highly-regulated environments requiring SOC2 compliance, AWS KMS / HashiCorp Vault integrations, centralized RBAC, and immutable Audit Logging, explore our Enterprise tier.  
> [Request Enterprise Demo](https://cal.com/rishabh-oberoi-apfkti/membrane-enterprise-demo-architecture-sync)

## Installation

```bash
pip install membrane-ai
```

## Quick Start
Membrane intercepts your prompt, deterministically masks the PII locally, and seamlessly restores it from the LLM's response. It is specifically built to handle LLM paraphrasing.

```python
from membrane import TrustLayer

layer = TrustLayer(provider="openai", api_key="sk-...")

# Membrane strips the PII, queries the LLM, and rehydrates the result automatically.
response = layer.call("My SSN is 123-45-6789. Send the report to john@example.com.")

print(response["final_response"]) 
```

## Features Breakdown

| Capability | Open Source (Free) | Enterprise |
|---------|-------------------|------------|
| **Core Anonymization** | Regex & Presidio NLP | Advanced Custom NER Models |
| **Provider Support** | OpenAI, Anthropic, Gemini, Ollama | Bring-Your-Own-Model (VPC) |
| **Key Management** | Environment Variables | AWS KMS, Azure KeyVault, HashiCorp Vault |
| **Audit Logging** | Local File-based | Datadog, Splunk, Immutable S3 Export |
| **Access Control** | N/A | Granular RBAC, SSO/SAML |
| **Scaling & State** | In-Memory (Single Node) | Distributed Redis Architecture |

## Why Membrane?

* **Zero Trust Security:** Sensitive data never leaves your infrastructure. Masking happens entirely locally before the outbound network request to external APIs (OpenAI/Anthropic) is even constructed. 
* **Near-Zero Latency:** Built for high-throughput streaming environments and rigorous production traffic without adding computational overhead.
* **Deterministic Context Preservation:** We don't just crudely mask data; we maintain contextual entity trackers. Even when an LLM radically paraphrases a response, we still resolve the identity deterministically and mathematically.
