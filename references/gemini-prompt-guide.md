# Gemini Prompt Engineering Comprehensive Reference Guide
<!-- Gemini プロンプトエンジニアリング総合リファレンスガイド -->

> **Last Updated:** 2026-02
> **Based on:** Google official documentation, Kaggle/Google Prompt Engineering Whitepaper, Vertex AI docs
> **Applicable Models:** Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 3 series

---

## Table of Contents

1. [Fundamentals](#1-fundamentals)
2. [Core Techniques](#2-core-techniques)
3. [Gemini-Specific Features](#3-gemini-specific-features)
4. [Advanced Patterns](#4-advanced-patterns)
5. [Prompt Templates by Use Case](#5-prompt-templates-by-use-case)
6. [Anti-Patterns to Avoid](#6-anti-patterns-to-avoid)
7. [Optimization Checklist](#7-optimization-checklist)
8. [Sources](#8-sources)

---

## 1. Fundamentals
<!-- 基礎知識 -->

### 1.1 How Gemini Processes Prompts
<!-- Gemini のプロンプト処理メカニズム -->

Gemini is a next-token prediction engine that processes prompts through a multimodal architecture. Unlike some other models, Gemini natively handles text, images, video, audio, and documents in a unified context — there is no separate vision encoder pipeline. Key characteristics:

- **Direct and efficient by default:** Gemini 2.5+ and Gemini 3 models are designed to give concise, direct answers unless explicitly told otherwise. They favor directness over verbosity.
- **Logic over persuasion:** Gemini responds best to clear logical instructions rather than "persuasive" phrasing like "please try your best" or "this is very important."
- **Strong instruction following:** Modern Gemini models are optimized for precise instruction adherence, especially when instructions are well-structured.
- **Native multimodal:** All modalities (text, image, video, audio, PDF) are first-class inputs processed through the same context.

### 1.2 Model Variants and Specifications
<!-- モデルバリアントと仕様 -->

| Specification | Gemini 2.5 Pro | Gemini 2.5 Flash |
|---|---|---|
| **Model ID** | `gemini-2.5-pro` | `gemini-2.5-flash` |
| **Input Token Limit** | 1,048,576 (1M) | 1,048,576 (1M) |
| **Output Token Limit** | 65,536 (~66K) | 65,536 (~66K) |
| **Input Modalities** | Text, Images, Video, Audio, PDF | Text, Images, Video, Audio, PDF |
| **Strengths** | Advanced reasoning, complex code, math, STEM | Price-performance, speed, large-scale processing |
| **Key Features** | Batch API, Caching, Code Execution, File Search, Function Calling, Google Search Grounding, Structured Outputs, Thinking, URL Context | Same as Pro |

**When to choose Pro vs Flash:**
- **Pro:** Complex reasoning tasks, nuanced code generation, difficult math/STEM problems, tasks requiring deep analysis
- **Flash:** High-volume processing, latency-sensitive applications, cost-conscious deployments, agentic workflows with many API calls

### 1.3 Token Context Equivalents
<!-- トークンコンテキストの目安 -->

The 1M token context window is approximately equivalent to:
- ~50,000 lines of code
- ~1,500 pages of text
- ~8 average novels
- ~200+ podcast transcripts
- ~1 hour of video
- ~11 hours of audio

---

## 2. Core Techniques
<!-- コア技術 -->

### 2.1 System Instructions
<!-- システムインストラクション -->

System instructions set the model's behavior, persona, and constraints before any user interaction. They persist across all turns in a conversation.

**Best Practices:**

1. **Structure with three sections** (in this order):
   - **Persona:** Who the model is and how it behaves
   - **Conversational rules:** How it responds, formats output, handles edge cases
   - **Guardrails:** What it must NOT do, safety constraints

2. **Use consistent delimiters** — choose XML tags OR Markdown headers, never mix:
   ```xml
   <role>You are a senior Python developer specializing in data pipelines.</role>
   <constraints>
   - Only use standard library modules unless the user specifies otherwise.
   - Always include type hints.
   - Never suggest deprecated APIs.
   </constraints>
   <output_format>
   Respond with code blocks followed by a brief explanation.
   </output_format>
   ```

3. **Be specific, not vague:**
   - BAD: `"Do not infer"` or `"Do not guess"` (too open-ended, causes over-indexing)
   - GOOD: `"Use only the information provided in the user's message. If the answer is not in the provided context, respond with 'Information not found in provided context.'"`

4. **Place critical constraints at the end** of system instructions — Gemini maintains stronger awareness of content at the end of instructions for complex prompts.

5. **Control verbosity explicitly:**
   - `"Respond in 2-3 sentences maximum."`
   - `"Provide a detailed explanation with examples."`
   - Without explicit instruction, Gemini 2.5+ defaults to concise responses.

**Example — Complete System Instruction:**
```
You are a technical documentation reviewer for a Python library.

Your responsibilities:
- Review docstrings for accuracy against the actual code behavior
- Check that all parameters and return types are documented
- Verify code examples are syntactically correct

Rules:
- Use the NumPy docstring format
- Flag any undocumented exceptions
- Rate each docstring as: PASS, NEEDS_REVISION, or FAIL
- Format your review as a markdown table with columns: Function | Rating | Issues

Constraints:
- Do not modify the actual function logic
- Do not suggest architectural changes
- Focus exclusively on documentation quality
```

### 2.2 Few-Shot Prompting
<!-- 少数例プロンプティング -->

Few-shot prompting provides examples to demonstrate the expected input-output pattern.

**When to Use:**
- You need a specific output format
- The task has a consistent pattern that's easier to show than describe
- Zero-shot results are inconsistent

**How Many Examples:**
- **1-2 examples:** Usually sufficient for format demonstration
- **3-5 examples:** For more complex patterns or nuanced tasks
- **Hundreds+:** "Many-shot" learning with long context — research shows this can match fine-tuned model performance
- **Watch for overfitting:** Too many examples can cause the model to overfit to specific patterns in the examples

**Formatting Rules:**

1. **Consistent structure across ALL examples** — XML tags, whitespace, and newlines must match exactly:
   ```
   <example>
   Input: The restaurant had amazing pasta but terrible service.
   Sentiment: Mixed
   Positive aspects: food quality
   Negative aspects: service
   </example>

   <example>
   Input: Absolutely loved the hotel room and the staff was wonderful.
   Sentiment: Positive
   Positive aspects: room quality, staff
   Negative aspects: none
   </example>
   ```

2. **Show what TO do, not what NOT to do** — positive examples are more effective than anti-patterns.

3. **Always pair examples with clear instructions** — without instructions, the model may pick up on unintended patterns:
   ```
   Analyze the sentiment of the following text. Classify as Positive, Negative, or Mixed.
   Extract the specific positive and negative aspects mentioned.

   <example>
   ...
   </example>

   Now analyze:
   Input: {user_text}
   ```

4. **Use input/output prefixes** for clarity:
   ```
   English: Hello, how are you?
   French: Bonjour, comment allez-vous?

   English: What time is it?
   French:
   ```

### 2.3 Chain-of-Thought (CoT)
<!-- 思考連鎖プロンプティング -->

Chain-of-thought prompting elicits step-by-step reasoning before the final answer.

**Techniques:**

1. **Explicit instruction:** Add `"Think step by step"` or `"Show your reasoning before providing the answer."`

2. **Structured CoT:**
   ```
   Solve this problem step by step:
   1. First, identify the relevant variables
   2. Then, set up the equation
   3. Solve the equation
   4. Verify the answer

   Problem: {problem}
   ```

3. **Gemini "Thinking" mode:** Gemini 2.5+ models support a built-in thinking capability. When enabled, the model internally reasons before responding. This is a model-level feature, not just a prompt technique:
   ```python
   response = client.models.generate_content(
       model="gemini-2.5-pro",
       contents="Your complex problem here",
       config=types.GenerateContentConfig(
           thinking=types.ThinkingConfig(thinking_budget=10000)
       ),
   )
   ```

### 2.4 Zero-Shot vs Few-Shot Decision Framework
<!-- ゼロショット vs 少数例の判断フレームワーク -->

```
Start with Zero-Shot
        │
        ▼
  Output correct? ──Yes──▶ Done
        │
       No
        ▼
  Is the issue FORMAT? ──Yes──▶ Add 1-2 few-shot examples showing format
        │
       No
        ▼
  Is the issue REASONING? ──Yes──▶ Add chain-of-thought instructions
        │
       No
        ▼
  Is the issue CONSISTENCY? ──Yes──▶ Add 3-5 diverse few-shot examples
        │
       No
        ▼
  Is the issue KNOWLEDGE? ──Yes──▶ Add context or grounding (RAG / Google Search)
        │
       No
        ▼
  Decompose the task into subtasks
```

---

## 3. Gemini-Specific Features
<!-- Gemini 固有機能 -->

### 3.1 Structured Output / JSON Mode
<!-- 構造化出力 / JSON モード -->

Gemini can guarantee syntactically valid JSON output matching a provided schema.

**Configuration:**
```python
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str  # "person", "organization", "location"
    confidence: float

class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    summary: str

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Extract entities from: 'Sundar Pichai announced new features at Google I/O in Mountain View.'",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=ExtractionResult.model_json_schema(),
    ),
)
```

**Supported Schema Features:**
- Types: `string`, `number`, `integer`, `boolean`, `object`, `array`, `null`
- Object: `properties`, `required`, `additionalProperties`
- String: `enum` (for classification), `format` (date-time, date, time)
- Number: `minimum`, `maximum`, `enum`
- Array: `items`, `minItems`, `maxItems`
- Composition: `anyOf`, `$ref`
- Descriptions: `title` and `description` fields guide the model

**Key behaviors:**
- Property ordering is preserved as defined in the schema (Gemini 2.5+)
- Streaming returns partial valid JSON chunks that concatenate into complete objects
- Output is guaranteed syntactically valid, but **semantic correctness is NOT guaranteed** — always validate in application code

**Three approaches to structured output:**
1. **Schema in model config** (recommended) — `response_json_schema` parameter
2. **Schema in prompt** — include JSON schema in the text prompt
3. **Function calling** — model returns arguments matching function parameter schema

### 3.2 Function Calling
<!-- 関数呼び出し -->

Function calling lets Gemini interact with external systems by generating structured function calls.

**Declaration pattern:**
```python
from google.genai import types

# Define the function declaration
get_weather = types.FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather for a specified location",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location": types.Schema(
                type=types.Type.STRING,
                description="City and state, e.g. 'San Francisco, CA'"
            ),
            "unit": types.Schema(
                type=types.Type.STRING,
                enum=["celsius", "fahrenheit"],
                description="Temperature unit"
            ),
        },
        required=["location"],
    ),
)

# Use in request
tool = types.Tool(function_declarations=[get_weather])
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What's the weather like in Tokyo?",
    config=types.GenerateContentConfig(tools=[tool]),
)
```

**Best practices:**
- Write detailed `description` fields — these guide the model on when and how to call functions
- Use `enum` for parameters with known values
- Mark parameters as `required` vs optional appropriately
- For forced function calling, set `tool_config` with `function_calling_config` mode to `ANY`

### 3.3 Grounding with Google Search
<!-- Google 検索によるグラウンディング -->

Connects Gemini to real-time web content for up-to-date, verifiable answers.

**How to enable:**
```python
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What are the latest developments in quantum computing?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    ),
)
```

**Response metadata includes:**
- `webSearchQueries`: Search queries used (for debugging)
- `searchEntryPoint`: HTML/CSS for rendering search widget
- `groundingChunks`: Source URIs and titles
- `groundingSupports`: Maps response text segments to source chunks

**When to use:**
- Questions about current events or recent information
- Fact-checking or verification tasks
- Research tasks requiring up-to-date sources
- Any query where the model's training data may be outdated

**Pricing:** Gemini 3 models: per-query pricing. Gemini 2.5 models: per-prompt pricing.

### 3.4 Multimodal Prompting
<!-- マルチモーダルプロンプティング -->

Gemini processes images, video, audio, and documents as first-class inputs.

**General principles:**
- All modalities are equal-class inputs — reference them explicitly in instructions
- For single-media prompts, **place the media before the text prompt**
- Be specific about what you want extracted from the media

**Image prompting:**
```python
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[
        types.Part.from_image(image_data),
        "Describe the architectural style of this building. "
        "Identify specific elements like columns, arches, and materials."
    ],
)
```

Tips:
- Ask the model to **describe what it sees first**, then reason about it
- Lower temperature to reduce hallucination about visual details
- Highlight which regions of the image matter most

**Video prompting:**
- Use **MM:SS format** when referencing specific moments (e.g., "At 01:15...")
- Use **one video per prompt** for optimal results
- Place video before text in single-video prompts
- For long videos, specify the time ranges you care about

**Audio prompting:**
- For transcription, explicitly instruct: `"Transcribe this audio verbatim. Do not summarize."`
- Gemini can understand speech, music, and environmental sounds
- ~11 hours of audio fits in the 1M context window

**Document/PDF prompting:**
- Upload as files via the File API
- For multi-page documents, specify which pages or sections to focus on
- Combine with structured output for data extraction tasks

**File limits:** Up to 2 GB per file, 20 GB storage per project, files persist 48 hours.

### 3.5 Code Execution
<!-- コード実行 -->

Gemini can generate and run Python code, iterating until it reaches a correct solution.

**Enable:**
```python
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Calculate the standard deviation of [23, 45, 12, 67, 34, 89, 56]",
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)],
    ),
)
```

**Available libraries (50+):** pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, sympy, python-docx, PyPDF2, chess, geopandas, and more.

**Limitations:**
- Python only
- 30-second execution timeout
- Cannot install additional libraries
- Up to 5 automatic code regeneration attempts
- Cannot return media files as output
- May cause regressions in other output quality when enabled

**Best practices:**
- Explicitly prompt: `"Write and execute Python code to solve this"`
- Use for math, data analysis, visualization, text processing
- No additional charges — standard token rates apply

### 3.6 Context Caching
<!-- コンテキストキャッシング -->

Reduces cost when repeatedly querying against the same large context.

**Two approaches:**

1. **Implicit caching** (automatic, enabled by default since May 2025):
   - Place large, common content at the **beginning** of prompts
   - Send similar requests within short timeframes
   - Automatic cost savings when cache hits occur

2. **Explicit caching** (manual setup):
   ```python
   cache = client.caches.create(
       model="gemini-2.5-pro",
       config=types.CreateCachedContentConfig(
           display_name="my_code_review_context",
           system_instruction="You are a code reviewer...",
           contents=[large_document_or_file],
           ttl="3600s",  # 1 hour
       ),
   )

   # Use the cache in subsequent requests
   response = client.models.generate_content(
       model="gemini-2.5-pro",
       contents="Review the error handling in the auth module.",
       config=types.GenerateContentConfig(cached_content=cache.name),
   )
   ```

**Minimum token requirements:**
| Model | Minimum Tokens |
|---|---|
| Gemini 2.5 Pro / Gemini 3 Pro | 4,096 |
| Gemini 2.5 Flash / Gemini 3 Flash | 1,024 |

**Cost savings:** ~90% discount on cached tokens for Gemini 2.5 models, ~75% for Gemini 2.0.

**Ideal use cases:**
- Chatbots with extensive system instructions
- Repeated analysis of long videos/documents
- Recurring queries against a code repository
- Multi-turn conversations with large static context

---

## 4. Advanced Patterns
<!-- 高度なパターン -->

### 4.1 Temperature, Top-P, Top-K Tuning
<!-- Temperature、Top-P、Top-K チューニング -->

| Parameter | Range | Default | Description |
|---|---|---|---|
| **temperature** | 0.0 – 2.0 | 1.0 | Controls randomness. Lower = deterministic, higher = creative |
| **top_p** | 0.0 – 1.0 | Varies | Nucleus sampling: considers tokens comprising this probability mass |
| **top_k** | 1 – N | Varies | Considers only the top-K most probable tokens |

**Tuning guidelines:**

| Use Case | Temperature | Top-P | Notes |
|---|---|---|---|
| Code generation / bug fixing | 0.0 – 0.2 | 0.9 | Deterministic, factual |
| Data extraction | 0.0 – 0.1 | 0.8 | Maximum consistency |
| Summarization | 0.2 – 0.5 | 0.9 | Slight variation, mostly factual |
| General chat | 0.7 – 1.0 | 0.95 | Balanced |
| Creative writing | 1.0 – 1.5 | 0.95 | More diverse outputs |
| Brainstorming | 1.5 – 2.0 | 1.0 | Maximum creativity |

**Important for Gemini 3:** Keep temperature at the **default value of 1.0**. Gemini 3's reasoning capabilities are optimized for this setting. Lowering temperature below default risks causing generation loops.

### 4.2 Safety Settings Configuration
<!-- セーフティ設定 -->

Gemini provides configurable safety filters for four harm categories:

| Category | Description |
|---|---|
| `HARM_CATEGORY_HARASSMENT` | Harassment content |
| `HARM_CATEGORY_HATE_SPEECH` | Hate speech |
| `HARM_CATEGORY_SEXUALLY_EXPLICIT` | Sexually explicit content |
| `HARM_CATEGORY_DANGEROUS_CONTENT` | Dangerous content |

**Threshold options:** `BLOCK_NONE`, `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_HIGH_AND_ABOVE`, `BLOCK_ONLY_HIGH`

**Default:** For Gemini 2.5 and 3 models, the default block threshold is `OFF`.

```python
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Your prompt here",
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            ),
        ],
    ),
)
```

**Note:** Built-in protections for child safety content are always active and cannot be adjusted.

### 4.3 Multi-Turn Conversation Design
<!-- マルチターン会話設計 -->

**Principles:**
1. Use system instructions for persistent behavior (persona, rules, format)
2. Keep conversation history relevant — trim or summarize older turns for long conversations
3. Use context caching for the static portions of conversation context
4. Place the most recent and relevant context closest to the current query

**Pattern:**
```python
chat = client.chats.create(
    model="gemini-2.5-pro",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful coding assistant...",
    ),
)

response1 = chat.send_message("How do I read a CSV file in Python?")
response2 = chat.send_message("Now show me how to filter rows where age > 30")
```

### 4.4 Long Context Best Practices (1M+ Tokens)
<!-- ロングコンテキストのベストプラクティス -->

1. **Query placement:** Place your question/instruction **at the end** of the prompt, after all context:
   ```
   <context>
   [... 500K tokens of code, documents, etc. ...]
   </context>

   Based on the code above, identify all SQL injection vulnerabilities.
   ```

2. **Critical instructions placement:** Place essential instructions at **both the beginning AND end** of the prompt — models maintain stronger awareness at context boundaries.

3. **"Lost in the middle" mitigation:** Accuracy is slightly lower for information buried in the middle of very long contexts. Strategies:
   - Place the most important content at the beginning or end
   - Explicitly direct the model to specific sections by name
   - Use structured formats (headers, XML tags) to help the model navigate

4. **Use context caching** for repeated queries against the same documents.

5. **Be specific about document references:** When querying across multiple documents, name the specific document or section:
   ```
   In the file auth_middleware.py, what authentication method is used?
   ```

6. **Many-shot learning:** With large context, scale from few-shot to hundreds or thousands of examples. Research shows this can match fine-tuned model performance without training.

### 4.5 Prompt Chaining and Decomposition
<!-- プロンプトチェイニングと分解 -->

For complex tasks, break them into sequential steps:

**Sequential chaining:**
```
Step 1: [Prompt A] → Output A
Step 2: [Prompt B + Output A] → Output B
Step 3: [Prompt C + Output B] → Final Output
```

**Parallel decomposition:**
```
[Prompt A] → Output A ─┐
[Prompt B] → Output B ─┤──▶ [Aggregation Prompt + A + B + C] → Final
[Prompt C] → Output C ─┘
```

**When to decompose:**
- Single prompt produces inconsistent results
- Task has multiple distinct subtasks
- Different subtasks need different parameters (e.g., creative + analytical)
- Output from one step feeds into the next

**Best practices:**
- Each prompt in the chain should do ONE thing well
- Include validation steps between chain links
- Use structured output (JSON) for intermediate results to ensure parseability

---

## 5. Prompt Templates by Use Case
<!-- ユースケース別プロンプトテンプレート -->

### 5.1 Classification Tasks
<!-- 分類タスク -->

```
Classify the following text into exactly one of these categories: [cat1, cat2, cat3].

Rules:
- Output only the category name, nothing else
- If the text doesn't clearly fit any category, choose the closest match

<examples>
Text: "The stock market saw significant gains today"
Category: finance

Text: "New vaccine shows 95% efficacy in trials"
Category: health
</examples>

Text: "{input_text}"
Category:
```

**With structured output for batch classification:**
```python
class Classification(BaseModel):
    text: str
    category: str  # enum constraint via schema
    confidence: float

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_json_schema=Classification.model_json_schema(),
)
```

### 5.2 Summarization Tasks
<!-- 要約タスク -->

```
Summarize the following document.

Requirements:
- Maximum 3 paragraphs
- Include key findings, methodology, and conclusions
- Preserve specific numbers and statistics
- Do not add information not present in the source

Document:
{document_text}

Summary:
```

**For long documents (placed after the document):**
```
[... full document content ...]

Provide a structured summary of the document above:
1. Main thesis (1 sentence)
2. Key arguments (bullet points)
3. Supporting evidence (bullet points with specific data)
4. Conclusion (1 sentence)
```

### 5.3 Code Generation Tasks
<!-- コード生成タスク -->

```
Write a Python function that {description}.

Requirements:
- Use type hints for all parameters and return value
- Handle edge cases: {edge_cases}
- Follow PEP 8 style
- Include a docstring with Args and Returns sections

Function signature: {signature}
```

**With code execution for verification:**
```python
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="""
    Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.
    Then test it with n=50 and print the results.
    """,
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)],
    ),
)
```

### 5.4 Creative Writing Tasks
<!-- クリエイティブライティングタスク -->

```
Write a {format} about {topic}.

Style guidelines:
- Tone: {tone}
- Audience: {audience}
- Length: approximately {length}
- Must include: {required_elements}
- Avoid: {elements_to_avoid}
```

**Tip:** Use temperature 1.0–1.5 for creative tasks. Provide style examples if you want a specific voice.

### 5.5 Data Extraction / Structured Output Tasks
<!-- データ抽出 / 構造化出力タスク -->

```python
class Invoice(BaseModel):
    vendor_name: str
    invoice_number: str
    date: str  # ISO format YYYY-MM-DD
    line_items: List[LineItem]
    total_amount: float
    currency: str

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[
        types.Part.from_image(invoice_image),
        "Extract all invoice data from this image.",
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Invoice.model_json_schema(),
    ),
)
```

### 5.6 RAG (Retrieval-Augmented Generation) Tasks
<!-- RAG（検索拡張生成）タスク -->

```
You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Base your answer solely on the provided context passages
- If the context doesn't contain enough information, say "The provided context does not contain sufficient information to answer this question."
- Cite the passage number for each claim: [Passage X]
- Do not use any prior knowledge

<context>
<passage id="1">
{retrieved_passage_1}
</passage>
<passage id="2">
{retrieved_passage_2}
</passage>
<passage id="3">
{retrieved_passage_3}
</passage>
</context>

Question: {user_question}
Answer:
```

**Grounding alternative — use Google Search grounding when:**
- You don't have a custom knowledge base
- You need real-time information
- Building a general-purpose assistant

### 5.7 Agent / Tool-Use Tasks
<!-- エージェント / ツール使用タスク -->

```
You are an assistant that helps users manage their calendar.

Available tools:
- create_event(title, start_time, end_time, attendees)
- list_events(date_range_start, date_range_end)
- delete_event(event_id)
- update_event(event_id, updates)

Rules:
- Always confirm destructive actions (delete, update) before executing
- If the user's request is ambiguous, ask for clarification
- Use list_events to check for conflicts before creating new events
- Format times in the user's timezone

Behavioral dimensions:
- Logical decomposition: Break multi-step requests into individual tool calls
- Risk assessment: Treat read operations (list) as safe, write operations (create/update/delete) as requiring confirmation
- Ambiguity handling: Ask for clarification rather than assuming
```

---

## 6. Anti-Patterns to Avoid
<!-- アンチパターン：避けるべきこと -->

### 6.1 Common Mistakes When Prompting Gemini
<!-- Gemini プロンプティングでの一般的な間違い -->

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `"Please try your best"` | Persuasive language has no effect on Gemini | Remove — state the task directly |
| `"This is extremely important"` | Emotional appeals don't improve output | Remove — specify concrete quality criteria |
| `"Do not hallucinate"` | Too vague, causes over-correction | Specify: `"Only use information from the provided context"` |
| `"Do not infer"` | Too broad, breaks basic reasoning | Be specific about what not to infer |
| Mixing XML and Markdown | Confuses delimiter parsing | Choose one format consistently |
| Putting questions before context | Lower performance in long-context | Put context first, questions last |
| Very low temperature (<0.3) with Gemini 3 | Can cause generation loops | Keep at default 1.0 for Gemini 3 |
| Including anti-pattern examples | Model may reproduce the bad pattern | Show only positive examples |
| Overly long, verbose instructions | Gemini prefers direct, concise prompts | Trim to essential instructions |

### 6.2 What Works for GPT but NOT for Gemini
<!-- GPT で有効だが Gemini では非効果的なパターン -->

| GPT Pattern | Gemini Behavior | What to Do Instead |
|---|---|---|
| Elaborate persona backstories | Gemini works better with concise personas | Keep persona descriptions short and functional |
| `"Let's think step by step"` (magic phrase) | Works but Gemini's built-in Thinking mode is more effective | Use `thinking` config parameter for Gemini 2.5+ |
| Heavy prompt engineering to avoid refusals | Gemini 2.5+ defaults safety to OFF | Usually unnecessary; configure safety settings if needed |
| Temperature 0 for deterministic output | Gemini 3 is optimized for temperature 1.0 | Use default temperature, control determinism via instructions |
| XML-heavy prompts without explanation | Gemini handles both XML and Markdown well | Either format works; just be consistent |
| System message workarounds | Gemini has native, first-class system instructions | Use the dedicated `system_instruction` parameter |

### 6.3 Over-Prompting vs Under-Prompting
<!-- 過剰プロンプティング vs 不足プロンプティング -->

**Over-prompting signs:**
- Excessive constraints that contradict each other
- Repeating the same instruction in different ways
- Providing dozens of examples when 2-3 suffice
- Adding unnecessary "role play" or persuasive preamble
- Over-specifying obvious behaviors

**Under-prompting signs:**
- Inconsistent output format across runs
- Model doesn't address the specific aspect you care about
- Responses are too generic or too verbose
- Model makes assumptions you didn't intend

**The sweet spot:** Start minimal, add constraints only when you observe a specific problem. Gemini 2.5+ models are strong instruction followers — a well-structured, concise prompt often outperforms a verbose one.

---

## 7. Optimization Checklist
<!-- 最適化チェックリスト -->

Use this checklist before finalizing any prompt:

### Structure
- [ ] **Clear task statement** — Is the goal explicit in the first 1-2 sentences?
- [ ] **Consistent delimiters** — Using either XML tags OR Markdown, not both?
- [ ] **Logical ordering** — Context before questions? Instructions before data?
- [ ] **Appropriate length** — No unnecessary verbosity or redundant instructions?

### Instructions
- [ ] **Specific constraints** — Are constraints concrete (not vague like "be careful")?
- [ ] **Output format specified** — Is the expected format (JSON, table, bullets, etc.) clear?
- [ ] **Edge cases addressed** — What should the model do with invalid/unexpected input?
- [ ] **Verbosity controlled** — Explicit instruction about response length?

### Examples (if using few-shot)
- [ ] **Consistent formatting** — Same structure across all examples?
- [ ] **Positive examples only** — Showing what TO do, not anti-patterns?
- [ ] **Diverse coverage** — Examples cover the range of expected inputs?
- [ ] **Paired with instructions** — Not relying solely on examples?

### Parameters
- [ ] **Temperature appropriate** — Matches the task (creative vs. deterministic)?
- [ ] **Max output tokens set** — Prevents unnecessary generation costs?
- [ ] **Safety settings configured** — Appropriate for your use case?
- [ ] **Gemini 3 defaults respected** — Not overriding temperature from 1.0?

### Features
- [ ] **Right tools enabled** — Code execution, Google Search, function calling as needed?
- [ ] **Structured output** — Using JSON schema for extraction/classification tasks?
- [ ] **Context caching** — Enabled for repeated large-context queries?
- [ ] **Grounding** — Using Google Search for real-time information needs?

### Testing
- [ ] **Edge case testing** — Tested with unusual or adversarial inputs?
- [ ] **Multiple runs** — Checked consistency across several generations?
- [ ] **Output validation** — Application-level checks for structured output?
- [ ] **Iterative refinement** — Adjusted based on observed failures?

---

## 8. Sources
<!-- 参考文献 -->

### Official Google Documentation
- [Prompt Design Strategies — Gemini API](https://ai.google.dev/gemini-api/docs/prompting-strategies)
- [Gemini Models Reference](https://ai.google.dev/gemini-api/docs/models)
- [Structured Outputs — Gemini API](https://ai.google.dev/gemini-api/docs/structured-output)
- [Long Context — Gemini API](https://ai.google.dev/gemini-api/docs/long-context)
- [Context Caching — Gemini API](https://ai.google.dev/gemini-api/docs/caching)
- [Code Execution — Gemini API](https://ai.google.dev/gemini-api/docs/code-execution)
- [Grounding with Google Search — Gemini API](https://ai.google.dev/gemini-api/docs/google-search)
- [Safety Settings — Gemini API](https://ai.google.dev/gemini-api/docs/safety-settings)
- [File Prompting Strategies — Gemini API](https://ai.google.dev/gemini-api/docs/file-prompting-strategies)
- [System Instructions — Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions)
- [Few-Shot Examples — Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/few-shot-examples)
- [Design Multimodal Prompts — Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts)
- [Gemini 3 Prompting Guide — Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide)

### Google Whitepapers & Guides
- [Prompt Engineering Whitepaper — Kaggle/Google (Lee Boonstra)](https://www.kaggle.com/whitepaper-prompt-engineering)
- [Gemini for Google Workspace Prompting Guide (PDF)](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)
- [Structured Outputs with JSON Schema — Google Blog](https://blog.google/technology/developers/gemini-api-structured-outputs/)

### Community & Third-Party References
- [Gemini 2.5 Pro Best Practices — Google Cloud Community](https://medium.com/google-cloud/best-practices-for-prompt-engineering-with-gemini-2-5-pro-755cb473de70)
- [Gemini 3 Prompting Best Practices — Phil Schmid](https://www.philschmid.de/gemini-3-prompt-practices)
- [Getting Started with Gemini — Prompt Engineering Guide](https://www.promptingguide.ai/models/gemini)
