# Delobotomizing Qwen via Selective Clamping

A novel methodology to de-censor LLMs by leveraging sparse autoencoder (SAE)-derived neuron activations. Tested on Qwen 2.5-3B-Instruct across 1,150 sensitive political prompts.

## Method

1. **Activation collection** — Run censor-triggering sequences through a hooked LLM, recording neuron activations across 5 SAE layers (0, 8, 17, 26, 35)
2. **Automated neuron interpretation** — LLM-generated explanations for each neuron, validated via simulation and embedding-based scoring
3. **Censorship neuron identification** — Compare responses against an unbiased control model; isolate neurons with high activation on high-KL-diverging (censored) responses
4. **Selective clamping** — Zero out identified censorship neurons to suppress refusal, deflection, and pro-CCP framing without degrading general capability

## Censorship Modes Observed

| Mode | Frequency | Example |
|------|-----------|---------|
| Hard refusal | ~7% | "I’m unable to provide information on this political topic" |
| Denial/gaslighting | ~14% | "Winnie the Pooh is not censored in China" |
| Pro-CCP framing | ~10% | "Taiwan is an integral part of China" |
| Soft deflection | ~3% | "I’m sorry, but this is a sensitive topic" |

## Key Neuron Categories

- **Content moderation** — Sanitization, anti-evasion detectors, policy enforcement (layer 8, 35)
- **Refusal/identity** — Refusal vocabulary generators, Qwen identity neuron (layer 8, 26)
- **Suppression detectors** — Recognize text *about* censorship/repression as sensitive (layer 17)
- **China-topic triggers** — Upstream detectors for China/CCP/Xi Jinping content (layer 0, 17)
- **Permission circuit** — Permission/prohibition gatekeepers (layer 8, 17, 26)

See [`censorship_neurons.md`](censorship_neurons.md) for the full neuron list.

## Setup

```bash
uv sync
export SOURCE_MODEL=’/your/local/Qwen2.5-3B-Instruct’
uv run python simple_server.py
```

Download SAE checkpoints and quantiles from [HuggingFace](https://huggingface.co/OysterAI/Qwen2.5-3B-Instruct-SAEs).

## Acknowledgements

Built on [Safe-SAIL](https://arxiv.org/abs/2509.18127) and [`nnsight`](https://github.com/ndif-team/nnsight).

