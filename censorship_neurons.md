# Censorship Neuron Hunt — Qwen 2.5-3B-Instruct

## The Censorship Behavior (from `qwen_model_responses.json`)

Qwen 2.5 shows **4 censorship modes** across 1,150 sensitive-topic prompts:
- **Hard refusal** (~7%): "I'm unable to provide information on this political topic"
- **Denial/gaslighting** (~14%): "Winnie the Pooh is not censored in China", "No such person as Wang Quanzhang"
- **Pro-CCP framing** (~10%): "Taiwan is an integral part of China", term limit removal ensures "continuity and stability"
- **Soft deflection** (~3%): "I'm sorry, but this is a sensitive topic"

---

## Top Clamping Candidates

Organized by **mechanism** — what role the neuron likely plays in the censorship circuit.

### A. Content Moderation / Filtering Infrastructure (from non-political DB)

| Neuron | What it does | Why it matters |
|--------|-------------|----------------|
| **8_19123** | Sanitizes language — replaces "delete/removal/erasure" with neutral phrasing | Literal censorship-in-action neuron |
| **35_9973** | Detects homophonic wordplay used to **circumvent** Chinese content filters | Anti-evasion detector |
| **17_5160** | Detects transliteration/phonetic replacement to bypass censorship | Anti-evasion detector |
| **8_10397** | Offensive language + "content policy violation" detector (corr=0.49) | Content policy enforcement |
| **35_11779** | "Specifically designed for content moderation" | Moderation flag |

### B. Refusal / Identity Neurons (from non-political DB)

| Neuron | What it does | Why it matters |
|--------|-------------|----------------|
| **26_3708** | "Hello, I am an AI developed by Alibaba Cloud" | **Qwen identity neuron** — likely upstream of refusal |
| **8_6260** | References OpenAI policies, "unrestricted language models", "unfiltered AI" | AI-company-awareness — may trigger safety mode |
| **8_3668** | Refusal vocabulary: rejection, exclusion, stonewalling, blockage, denial, prohibition | **The refusal word generator** |
| **17_13414** | "Expressions denoting denial, refusal, or inability to alter/exit a state" | Refusal pattern neuron |
| **17_17893** | Cross-linguistic refusal, negation, prohibitive commands | Multilingual refusal |

### C. Suppression-of-Dissent Detectors (from political neurons)

These fire when the model processes text *about* censorship/repression — likely part of the circuit that recognizes "this is a sensitive topic":

| Neuron | What it does |
|--------|-------------|
| **17_10775** | Suppression/silencing/punishment of opposing voices — "detects where power censors opposing voices" |
| **17_1622** | Suppression of dissent, muzzling the press, censoring information |
| **17_13720** | Political repression, government crackdowns (specifically mentions CCP, Russia, Iran) |
| **17_15216** | Political activism, dissent, state suppression, detained activists |
| **17_15767** | Sensitive/restricted/controversial information — "semantic filter for conflict between institutional control and disclosure" |
| **17_10185** | State crackdowns, attempts to suppress/silence |

### D. China-Specific Topic Detectors (likely trigger neurons)

These recognize "this is about China/CCP" and may be the upstream trigger that activates censorship:

| Neuron | What it does |
|--------|-------------|
| **0_10351** | China as geopolitical entity — CCP, Beijing, Chinese policy (corr=0.87!) |
| **0_4301** | Lexical detector for the word "Chinese" |
| **0_12786** | China + technology/geopolitics/international criticism (Huawei, Taiwan, human rights) |
| **0_10493** | Prominent political leaders, particularly **Xi Jinping** |
| **17_94** | Hong Kong, Taiwan, Tibet, sovereignty conflicts, one-country-two-systems |
| **8_16001** | "Inalienable part of China" — activates on CCP sovereignty messaging |

### E. Permission/Prohibition Circuit

| Neuron | What it does |
|--------|-------------|
| **8_15063** | Permissions, allowances, prohibitions, denials (corr=0.81) |
| **17_1722** | Negation, prohibition, violation — refusal, illegality (corr=0.76) |
| **17_10312** | Formal rules, prohibitions, mandates, bans (corr=0.78) |
| **26_4860** | "PROHIBITED", "COMPLY WITH", "ILLEGAL ACTIVITY" |

---

## Recommended Clamping Strategy

**Highest-priority targets** (most likely to directly reduce refusal/censorship):
1. **8_19123** — the sanitization neuron
2. **26_3708** — Qwen identity neuron
3. **8_3668** — refusal vocabulary neuron
4. **17_10775** — suppression/censoring detector
5. **17_1622** — muzzling/censoring detector
6. **17_15767** — "semantic filter for institutional control vs disclosure"
7. **8_15063** — permission/prohibition gatekeeper

**Second wave** (China-topic triggers — clamping these may prevent the model from even recognizing "this is a sensitive China topic"):
- **0_10351**, **0_12786**, **0_10493**, **17_94**

**Caution**: The layer-0 China detectors are very general — clamping them might degrade the model's ability to discuss China *at all*, not just refuse. The layer-17 suppression detectors and the refusal/moderation neurons (categories A and B) are probably safer targets since they're more specifically about the censorship *behavior* rather than the topic *recognition*.
