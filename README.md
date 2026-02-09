# gemini-prompt-eng

Google Gemini モデル向けに最適化されたプロンプトを生成する Claude Code Skill です。

ユーザーの目的をヒアリングし、最適なプロンプトパターンを選択して、パラメータ設定・API コード例付きのプロダクションレディなプロンプトを出力します。

## 対応モデル

- Gemini 3 Pro / 3 Flash
- Gemini 2.5 Pro / 2.5 Flash

## 機能

- **対話型ヒアリング** - ゴール・モデル・ユースケース・出力形式を段階的にヒアリング
- **パターン自動選択** - Zero-shot / Few-shot / Chain-of-Thought / Structured Output / Multimodal / Function Calling / Grounding から最適なパターンを選択
- **モデル別パラメータ最適化** - Gemini 3 と 2.5 の仕様差異を反映した推奨パラメータを設定
- **API コード例生成** - `google.genai` SDK を使った Python コードを出力
- **品質チェックリスト** - 生成前にベストプラクティスとの適合を自動検証
- **多言語対応** - ユーザーの言語に合わせて応答（日本語 / 英語）

## インストール

このディレクトリを Claude Code プロジェクトの `.claude/skills/` にコピーしてください。

```bash
cp -r gemini-prompt-eng /path/to/your-project/.claude/skills/
```

配置後のディレクトリ構造:

```
your-project/
└── .claude/
    └── skills/
        └── gemini-prompt-eng/
            ├── SKILL.md                          # スキル定義（必須）
            └── references/
                └── gemini-prompt-guide.md         # 詳細リファレンス
```

## 使い方

Claude Code で以下のようなフレーズを入力するとスキルが起動します。

```
Geminiのプロンプトを書いて
Write a prompt for Gemini
Geminiプロンプト最適化
Gemini prompt engineering
Gemini向けのプロンプトを最適化して
Gemini用のシステム指示を作って
```

スラッシュコマンドからも呼び出せます:

```
/gemini-prompt-eng
```

### ワークフロー

1. **ヒアリング** - ゴールとモデルを確認し、必要に応じてユースケース・出力形式を質問
2. **パターン選択** - ヒアリング結果から最適なプロンプトパターンを決定
3. **ベストプラクティス適用** - モデル固有の推奨事項を反映
4. **プロンプト生成** - System Instruction / User Prompt / Few-Shot Examples / 推奨パラメータ / API コード例を一括出力
5. **設計判断の説明** - なぜそのパターン・パラメータを選んだかを解説

### 出力例

```
## Generated Prompt

### System Instruction
You are a customer feedback classifier. Classify each feedback message into
exactly one category and extract the key topic.

### Recommended Parameters
- Model: gemini-3-pro
- Temperature: 1.0 (Gemini 3 default)
- Thinking level: low
- Response format: application/json

### API Code Example
[Python code using google.genai]
```

## Gemini 3 の注意点

このスキルは Gemini 3 の重要な仕様変更を反映しています:

| 項目 | Gemini 2.5 | Gemini 3 |
|------|-----------|----------|
| Temperature | タスクに応じて 0.0-1.0 | **常に 1.0**（下げるとループ発生） |
| 思考制御 | `thinking_budget` (トークン数) | `thinking_level` (low/high) |
| プロンプト長 | 長い指示も有効 | **短く直接的な指示が最適** |
| 出力スタイル | 設定次第 | デフォルトで簡潔 |

## スキル構成

```
gemini-prompt-eng/
├── SKILL.md                          # スキル本体（手順・テンプレート・パラメータガイド）
└── references/
    └── gemini-prompt-guide.md        # 公式ベストプラクティスの詳細リファレンス（~950行）
```

- **SKILL.md** - ヒアリング手順、パターン選択ロジック、6 種のプロンプトテンプレート、パラメータガイド、品質チェックリスト
- **references/gemini-prompt-guide.md** - Gemini の公式ドキュメントに基づく詳細なプロンプト技法の解説。SKILL.md から参照される

## ライセンス

MIT
