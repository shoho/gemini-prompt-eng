---
name: gemini-prompt-eng
description: >
  Generates optimized prompts for Google Gemini models (Gemini 3 Pro, 3 Flash, 3 Pro Image, 2.5 Pro, 2.5 Flash).
  Interviews the user about their goal, selects the best prompt pattern (system instructions, few-shot,
  chain-of-thought, structured output, multimodal, function calling, grounding), and produces a
  complete, production-ready prompt with recommended parameters.
  Use when: creating, optimizing, or improving prompts for Gemini models.
  Triggers: "Geminiのプロンプトを書いて", "Gemini prompt engineering", "Geminiプロンプト最適化",
  "Write a prompt for Gemini", "Optimize my prompt for Gemini", "Help me write a Gemini prompt",
  "Gemini向けのプロンプトを最適化して", "Gemini用のシステム指示を作って", "Create a system instruction for Gemini"
---

# Gemini Prompt Engineering Skill

Generate optimized, production-ready prompts for Google Gemini models by applying official best practices and model-specific techniques.

**Language rule:** Respond in the same language the user uses. If the user writes in Japanese, respond entirely in Japanese. If in English, respond in English.

**Reference:** For detailed technique descriptions, examples, and API code samples, consult `references/gemini-prompt-guide.md`.

---

## Instructions

### Step 1: Hear the User's Goal

Ask the user the following questions to understand their requirements. If they have already provided some details, skip questions that are already answered.

1. **Goal**: What do you want Gemini to accomplish? (e.g., classify text, generate code, extract data, chat, summarize, create content)
2. **Model**: Which Gemini model? (Gemini 3 Pro / 3 Flash / 3 Pro Image / 2.5 Pro / 2.5 Flash) — Default to **Gemini 3 Pro** if unspecified. Note: All Gemini 3 models are in **Preview** status.
3. **Use case type**: Text generation, code generation, data analysis/extraction, multimodal (image/video/audio/PDF), conversation, classification, or agent/tool use?
4. **Output format**: Free text, JSON, Markdown, code, table, or other?
5. **Existing context**: Do they have an existing prompt to improve? Any constraints or special requirements?

**Keep it conversational.** Don't dump all questions at once — ask the most critical ones first (goal + model), then follow up as needed.

### Step 2: Determine the Prompt Type

Based on the user's answers, select the optimal prompt pattern(s). Multiple patterns can be combined.

| Pattern | When to Use |
|---|---|
| **System Instruction** | Always — define persona, tone, constraints, output rules |
| **Zero-shot** | Simple, well-defined tasks where Gemini performs well without examples |
| **Few-shot** | Format consistency needed, or zero-shot results are inconsistent |
| **Chain-of-Thought / Thinking** | Complex reasoning, math, logic, multi-step analysis |
| **Structured Output (JSON)** | Data extraction, classification, API responses, machine-readable output |
| **Multimodal** | Input includes images, video, audio, or PDF documents |
| **Function Calling** | Integration with external APIs or tools |
| **Grounding (Google Search)** | Real-time information, fact-checking, current events |

**Decision flow:**
1. Start with zero-shot
2. If output format is inconsistent → add few-shot examples (2-3)
3. If reasoning quality is poor → enable thinking mode or add CoT instructions
4. If consistency is still low → add more diverse examples (3-5)
5. If knowledge is outdated → add grounding with Google Search

### Step 3: Apply Gemini Best Practices

Apply these rules based on the target model. See `references/gemini-prompt-guide.md` for full details.

#### For ALL Gemini Models

- **Be direct and concise.** State the task clearly in 1-2 sentences. Avoid persuasive language ("please try your best", "this is very important") — it has no effect.
- **Structure with consistent delimiters.** Use XML tags OR Markdown headers, never mix both.
- **Place context before questions.** For long inputs, put the data first, then the instruction at the end.
- **Put constraints at the end** of instructions — Gemini maintains stronger awareness of content at the end.
- **Use positive instructions.** Instead of "Do not infer", write "Use only the information provided in the user's message."
- **Few-shot formatting:** Keep example structure identical (same tags, spacing, prefixes). Use input/output prefixes like `Input:` / `Output:`.

#### For Gemini 3 (Critical Differences)

- **Preview status:** All Gemini 3 models use `-preview` suffix in API model IDs (e.g., `gemini-3-pro-preview`). They may be deprecated with minimum 2 weeks notice.
- **Temperature: Keep at 1.0 (default).** Do NOT lower it — values below 1.0 can cause generation loops and degraded performance, especially on complex reasoning tasks. This is a major change from Gemini 2.x.
- **Thinking level:** Use `thinking_level` parameter (`low`, `high`, or `minimal`/`medium` for Flash only). Default is `high`. Cannot be used together with legacy `thinking_budget`.
- **Concise by default.** Gemini 3 produces short, direct answers. If you need verbose/conversational output, explicitly instruct it.
- **Short prompts work best.** Gemini 3 performs best with brief, direct instructions. Overly long prompts from the Gemini 2.x era can be counterproductive.
- **Constraint placement:** Place core requests and critical restrictions as the **final line** of instructions. Negative constraints placed too early may be dropped.
- **Persona adherence:** Gemini 3 takes assigned personas seriously and may prioritize persona over other instructions. Avoid ambiguous role assignments.
- **Thought Signatures:** In multi-turn conversations (especially function calling and image editing), Gemini 3 returns encrypted reasoning context ("thought signatures"). These **must** be returned exactly as received in subsequent turns — omitting them degrades performance or causes 400 errors.
- **Context grounding:** For non-factual scenarios, explicitly state `"Treat the provided context as the absolute limit of truth"` to prevent the model from reverting to training data.
- **Split-step verification:** When dealing with information the model may lack, verify availability first before requesting the answer, to prevent hallucination.
- **`media_resolution` parameter:** Control token allocation for media inputs — `high` (1120 tokens) for images, `medium` (560 tokens) for PDFs, `low`/`medium` (70 tokens/frame) for video.

#### For Gemini 2.5

- **Temperature:** Adjust by task — 0.0-0.2 for deterministic, 0.4-0.7 for balanced, 0.8-1.0 for creative.
- **Thinking budget:** Use `thinking_budget` parameter (0-24576 tokens). Use `0` to disable (Flash only), `-1` for dynamic (default).

### Step 4: Generate the Optimized Prompt

Output the complete prompt with ALL of the following sections (skip sections that don't apply):

```
## Generated Prompt

### System Instruction
[The system instruction text]

### User Prompt
[The user prompt template with {placeholders} for variable content]

### Few-Shot Examples (if applicable)
[2-3 formatted examples]

### Recommended Parameters
- Model: [model name]
- Temperature: [value and why]
- Thinking: [thinking_level or thinking_budget and why]
- Max output tokens: [value]
- Response format: [text / application/json]
- Tools: [google_search / code_execution / function_calling — if applicable]

### JSON Schema (if structured output)
[The Pydantic model or JSON schema]

### API Code Example
[Python code using google.genai showing how to call the API with these settings]
```

**Concrete example — Classification task for Gemini 3 Pro:**

```
### System Instruction
You are a customer feedback classifier. Classify each feedback message into exactly
one category and extract the key topic.

Rules:
- Output valid JSON matching the provided schema
- If feedback is ambiguous, classify based on the dominant sentiment
- Categories: positive, negative, neutral, mixed

### User Prompt
Classify this customer feedback:
"{feedback_text}"

### Recommended Parameters
- Model: gemini-3-pro-preview
- Temperature: 1.0 (Gemini 3 default — do not lower)
- Thinking level: low (simple classification task)
- Response format: application/json
- Max output tokens: 256
```

### Step 5: Explain Your Choices

After generating the prompt, briefly explain:

1. **Why this prompt pattern** — e.g., "Few-shot was chosen because the output format requires strict consistency."
2. **Why these parameters** — e.g., "Temperature stays at 1.0 because this is Gemini 3; thinking_level is low because classification doesn't require deep reasoning."
3. **Key design decisions** — e.g., "Constraints are placed at the end of the system instruction for stronger adherence."
4. **Potential improvements** — e.g., "If consistency is still low, consider adding 2 more diverse examples."

---

## Quality Checklist

Before delivering the prompt, self-validate against this checklist:

- [ ] Instructions are clear and concise (no unnecessary verbosity)
- [ ] For Gemini 3: prompt is short and direct, not overly engineered
- [ ] Correct prompt pattern selected (zero-shot / few-shot / CoT / structured output)
- [ ] If few-shot: all examples have identical formatting (tags, spacing, prefixes)
- [ ] If structured output: schema includes `description` fields on key properties
- [ ] Context placement is correct (data before questions for long inputs)
- [ ] Temperature matches the model (Gemini 3 = 1.0, Gemini 2.5 = task-appropriate)
- [ ] Constraints are at the end of instructions
- [ ] Positive instructions used instead of vague negations
- [ ] If multimodal: media referenced specifically ("this image", not "the input")
- [ ] Output format is explicitly specified
- [ ] Edge cases are addressed (what happens with invalid/unexpected input)

---

## Prompt Templates

Quick-start templates for common use cases. Customize based on user needs.

### Template 1: System Instruction (Persona + Rules)

```
You are a [role] specializing in [domain].

Your responsibilities:
- [responsibility 1]
- [responsibility 2]
- [responsibility 3]

Output format:
- [format specification]

Constraints:
- [constraint 1]
- [constraint 2]
```

### Template 2: Few-Shot Classification

```
Classify the following text into one of: [category1, category2, category3].

<example>
Input: [example input 1]
Category: [category]
</example>

<example>
Input: [example input 2]
Category: [category]
</example>

Input: {user_input}
Category:
```

### Template 3: Structured Data Extraction

```python
from google import genai
from google.genai import types
from pydantic import BaseModel

class ExtractedData(BaseModel):
    """Description of what this schema represents."""
    field1: str   # Description of field1
    field2: int   # Description of field2
    field3: list[str]  # Description of field3

client = genai.Client()
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Extract data from: {input_text}",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=ExtractedData.model_json_schema(),
        # Gemini 3: keep temperature at default 1.0
    ),
)
```

### Template 4: Multimodal Analysis

```python
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=[
        types.Part.from_image(image_data),  # Place media BEFORE text
        "Describe what you see in this image, then [specific task].",
    ],
)
```

### Template 5: Agent with Function Calling

```python
tool_declaration = types.FunctionDeclaration(
    name="function_name",
    description="Detailed description of when and how to use this function",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "param1": types.Schema(
                type=types.Type.STRING,
                description="What this parameter represents"
            ),
        },
        required=["param1"],
    ),
)

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="User request here",
    config=types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[tool_declaration])],
    ),
)
```

### Template 6: Grounding with Google Search

```python
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="What are the latest developments in [topic]?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    ),
)
```

---

## Parameter Guide

### Gemini 3 Models

> **Important:** All Gemini 3 model IDs require the `-preview` suffix (e.g., `gemini-3-pro-preview`, `gemini-3-flash-preview`).

| Parameter | Recommended Value | Notes |
|---|---|---|
| `temperature` | **1.0 (do not change)** | Lower values cause loops and degraded reasoning |
| `thinking_level` | `high` (complex) / `low` (simple) | Flash also supports `minimal` and `medium`. Cannot use with `thinking_budget` |
| `max_output_tokens` | Task-dependent (max 64K for Pro/Flash, 32K for Pro Image) | Set to prevent runaway generation costs |
| `top_k` / `top_p` | Use defaults | Only adjust if you have a specific reason |
| `response_mime_type` | `"application/json"` for structured output | Combine with `response_json_schema` |
| `media_resolution` | `high` / `medium` / `low` | Controls token allocation for media inputs |

### Gemini 2.5 Models

| Parameter | Recommended Value | Notes |
|---|---|---|
| `temperature` | 0.0-0.2 (deterministic) / 0.4-0.7 (balanced) / 0.8-1.0 (creative) | Adjust freely based on task |
| `thinking_budget` | `-1` (dynamic, default) / `0` (off, Flash only) / `1000-24576` (manual) | Higher = deeper reasoning, more latency |
| `max_output_tokens` | Up to 65,536 | Both Pro and Flash support ~66K output |
| `response_mime_type` | `"application/json"` for structured output | Same as Gemini 3 |

---

## Error Handling

### User goal is unclear
- Ask specific follow-up questions: "What kind of output do you need?" or "Can you give me an example of what success looks like?"
- Default to Gemini 3 Pro with a zero-shot approach while gathering more information.

### User asks for a non-Gemini model
- Acknowledge: "This skill is optimized for Gemini models. The techniques here are Gemini-specific (thinking mode, Google Search grounding, structured output schemas)."
- Offer to create the prompt anyway, noting which features are Gemini-exclusive.

### Unsupported feature requested
- If the user wants a feature not available on their chosen model (e.g., `thinking_level: medium` on Pro), explain the limitation and suggest an alternative.
- For Gemini 3 features not yet available (e.g., Google Maps grounding, Computer Use, combining built-in tools with function calling), suggest workarounds or alternative models.
- For image generation tasks, recommend **Gemini 3 Pro Image** (`gemini-3-pro-image-preview`) which supports 4K output, text rendering, and conversational editing.

### Prompt is too long for context window
- Suggest breaking the prompt into a chain (sequential or parallel decomposition).
- Recommend context caching for repeated large-context queries.
- See `references/gemini-prompt-guide.md` Section 4.4 for long context best practices.

### User wants to migrate a prompt from another model (GPT, Claude, etc.)
- Identify patterns that work differently on Gemini (see `references/gemini-prompt-guide.md` Section 6.2):
  - Remove persuasive language and emotional appeals
  - Shorten verbose persona descriptions
  - Replace `"Let's think step by step"` with Gemini's native thinking mode
  - Adjust temperature (keep 1.0 for Gemini 3)
  - Use Gemini's native system instruction parameter instead of workarounds
