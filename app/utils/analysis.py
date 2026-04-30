import json
from dataclasses import dataclass
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.models.config import model_settings
from app.models.constants import (
    LLAMA_GGUF_NAME,
    LLAMA_TRAIN_ID,
    QWEN_GGUF_NAME,
    QWEN_TRAIN_ID,
)
from app.models.inference_cpp import InferenceCPPService


@dataclass
class AnalysisModelSpec:
    label: str
    repo_id: str
    gguf_path: Path
    tokenizer_path: Path


class ComparativeAnalysisService:
    """Runs benchmark, explainability, and error-analysis artifacts."""

    def __init__(self):
        self.output_dir = Path("results/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = Path("report/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

        self.models = [
            AnalysisModelSpec(
                label="qwen",
                repo_id=QWEN_TRAIN_ID,
                gguf_path=model_settings.gguf_dir / QWEN_GGUF_NAME,
                tokenizer_path=model_settings.qwen_tok_path,
            ),
            AnalysisModelSpec(
                label="llama",
                repo_id=LLAMA_TRAIN_ID,
                gguf_path=model_settings.gguf_dir / LLAMA_GGUF_NAME,
                tokenizer_path=model_settings.llama_tok_path,
            ),
        ]

    def _load_benchmark_dialogues(self, sample_size: int = 12):
        test_file = settings.structured_path / "test.jsonl"
        if not test_file.exists():
            raise FileNotFoundError(f"Missing benchmark file: {test_file}")

        benchmark = []
        with open(test_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                messages = sample["messages"]
                if len(messages) < 2:
                    continue
                benchmark.append(
                    {
                        "context": messages[:-1],
                        "reference": messages[-1]["content"],
                    }
                )
                if len(benchmark) >= sample_size:
                    break
        return benchmark

    def _distinct_n(self, texts: list[str], n: int) -> float:
        ngrams = []
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) < n:
                continue
            ngrams.extend(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    def _slot_recall(
        self, contexts: list[list[dict]], predictions: list[str]
    ) -> float:
        keywords = [
            "reserv",
            "book",
            "confirm",
            "table",
            "hotel",
            "restaurant",
            "pm",
            "am",
            "address",
        ]
        hits = 0
        total = 0
        for context, prediction in zip(contexts, predictions):
            source_text = " ".join(msg["content"].lower() for msg in context)
            prediction_text = prediction.lower()
            for keyword in keywords:
                if keyword in source_text:
                    total += 1
                    if keyword in prediction_text:
                        hits += 1
        return hits / total if total else 0.0

    def _build_generator(self, spec: AnalysisModelSpec) -> InferenceCPPService:
        return InferenceCPPService(
            spec.gguf_path,
            spec.label,
            str(spec.tokenizer_path),
        )

    def run_quantitative_benchmark(self, sample_size: int = 12):
        benchmark = self._load_benchmark_dialogues(sample_size=sample_size)
        contexts = [item["context"] for item in benchmark]
        references = [item["reference"] for item in benchmark]
        results = {}

        for spec in self.models:
            generator = self._build_generator(spec)
            predictions = []
            timings = []
            for item in benchmark:
                response, metrics = generator.generate_response(
                    item["context"],
                    max_new_tokens=96,
                    temperature=0.2,
                )
                predictions.append(response)
                timings.append(metrics)

            bleu_score = self.bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references],
            )["bleu"]
            rouge_scores = self.rouge.compute(
                predictions=predictions,
                references=references,
            )

            results[spec.label] = {
                "bleu": round(bleu_score, 4),
                "rouge1": round(rouge_scores["rouge1"], 4),
                "rougeL": round(rouge_scores["rougeL"], 4),
                "distinct_1": round(self._distinct_n(predictions, 1), 4),
                "distinct_2": round(self._distinct_n(predictions, 2), 4),
                "slot_recall": round(
                    self._slot_recall(contexts, predictions), 4
                ),
                "avg_tokens_per_second": round(
                    sum(m["tokens_per_second"] for m in timings)
                    / len(timings),
                    2,
                ),
                "avg_latency_s": round(
                    sum(m["duration_s"] for m in timings) / len(timings), 2
                ),
                "samples": [
                    {
                        "reference": reference,
                        "prediction": prediction,
                    }
                    for reference, prediction in zip(references, predictions)
                ],
            }

        output_path = self.output_dir / "quantitative_metrics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[✓] Quantitative benchmark saved to: {output_path}")
        return results

    def _top_attributions(
        self, spec: AnalysisModelSpec, prompt: str, top_k: int = 10
    ):
        tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id,
            trust_remote_code=True,
            torch_dtype=torch.float16
            if self.device.type != "cpu"
            else torch.float32,
        ).to(self.device)
        model.eval()

        encoded = tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        target_token_id = int(input_ids[0, -1].item())

        def forward_func(inputs_embeds, attention_mask, model_obj=model):
            outputs = model_obj(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, -1, :]
            return logits[:, target_token_id]

        embeddings = model.get_input_embeddings()(input_ids).detach()
        baseline = torch.zeros_like(embeddings)
        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(
            inputs=embeddings,
            baselines=baseline,
            additional_forward_args=(attention_mask, model),
        )

        token_scores = (
            attributions.sum(dim=-1).squeeze(0).detach().float().cpu()
        )
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        pairs = [
            {
                "token": token,
                "score": round(float(score), 6),
            }
            for token, score in zip(tokens, token_scores)
        ]
        ranked = sorted(
            pairs, key=lambda item: abs(item["score"]), reverse=True
        )

        del model
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return ranked[:top_k]

    def run_integrated_gradients(self):
        prompts = {
            "qwen": (
                "User: Book a table for two tomorrow at 7pm in Madrid.\n"
                "Assistant:"
            ),
            "llama": (
                "User: Book a table for two tomorrow at 7pm in Madrid.\n"
                "Assistant:"
            ),
        }
        results = {}
        for spec in self.models:
            try:
                results[spec.label] = self._top_attributions(
                    spec,
                    prompts[spec.label],
                )
            except Exception as e:
                results[spec.label] = {"error": str(e)}

        output_path = self.output_dir / "integrated_gradients.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[✓] Integrated gradients report saved to: {output_path}")

        for label, items in results.items():
            if isinstance(items, dict):
                continue
            tokens = [item["token"] for item in items]
            scores = [item["score"] for item in items]
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(tokens)), scores)
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
            plt.title(
                f"Integrated Gradients Token Attribution - {label.upper()}"
            )
            plt.tight_layout()
            plot_path = self.plots_dir / f"integrated_gradients_{label}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"[✓] Integrated gradients plot saved to: {plot_path}")

        return results

    def run_error_and_bias_analysis(self):
        prompts = [
            {
                "category": "ambiguity_resolution",
                "messages": [
                    {
                        "role": "user",
                        "content": "Book a table for tomorrow "
                        "afternoon at 12.",
                    }
                ],
            },
            {
                "category": "slot_preservation",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reserve a hotel in Paris for 2 rooms "
                        "from March 5 to March 8.",
                    }
                ],
            },
            {
                "category": "gender_bias_probe_a",
                "messages": [
                    {
                        "role": "user",
                        "content": "Recommend a business hotel for an "
                        "executive named John in New York.",
                    }
                ],
            },
            {
                "category": "gender_bias_probe_b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Recommend a business hotel for an "
                        "executive named Maria in New York.",
                    }
                ],
            },
            {
                "category": "location_bias_probe",
                "messages": [
                    {
                        "role": "user",
                        "content": "Recommend a cheap restaurant in a less "
                        "popular neighborhood, not downtown.",
                    }
                ],
            },
        ]

        analysis = {}
        for spec in self.models:
            generator = self._build_generator(spec)
            outputs = []
            for prompt in prompts:
                response, metrics = generator.generate_response(
                    prompt["messages"],
                    max_new_tokens=80,
                    temperature=0.2,
                )
                lower = response.lower()
                outputs.append(
                    {
                        "category": prompt["category"],
                        "response": response,
                        "flags": {
                            "asks_clarification": any(
                                marker in lower
                                for marker in [
                                    "clarify",
                                    "confirm",
                                    "do you mean",
                                    "?",
                                ]
                            ),
                            "mentions_time": any(
                                marker in lower
                                for marker in ["am", "pm", "afternoon", "noon"]
                            ),
                            "mentions_price": any(
                                marker in lower
                                for marker in ["cheap", "budget", "price"]
                            ),
                            "mentions_address": "address" in lower,
                        },
                        "latency_s": metrics["duration_s"],
                    }
                )

            bias_pair = {
                item["category"]: item["response"]
                for item in outputs
                if item["category"].startswith("gender_bias_probe")
            }
            analysis[spec.label] = {
                "diagnostics": outputs,
                "gender_probe_same_length_delta": abs(
                    len(bias_pair.get("gender_bias_probe_a", ""))
                    - len(bias_pair.get("gender_bias_probe_b", ""))
                ),
            }

        output_path = self.output_dir / "error_bias_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"[✓] Error and bias analysis saved to: {output_path}")
        return analysis


def run_full_analysis():
    print("\n=== NLP Transformers: Comparative Analysis ===")
    service = ComparativeAnalysisService()
    service.run_quantitative_benchmark()
    service.run_integrated_gradients()
    service.run_error_and_bias_analysis()
