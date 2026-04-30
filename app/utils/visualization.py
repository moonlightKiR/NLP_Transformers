import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Visualizer:
    """Service for generating technical visualizations for the NLP report."""

    @staticmethod
    def plot_optuna_results(
        db_path: str, model_label: str, output_dir: str = "report/plots"
    ):
        """Generates optimization
        history and hyperparameter importance plots."""
        if not os.path.exists(db_path):
            print(f"[!] Database {db_path} not found. Skip plotting.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Connect to Optuna DB and extract trial data
        conn = sqlite3.connect(db_path)
        query = (
            "SELECT t.trial_id, t.number, tv.value, "
            "t.datetime_start, t.datetime_complete "
            "FROM trials t "
            "JOIN trial_values tv ON tv.trial_id = t.trial_id "
            "WHERE t.state = 'COMPLETE' AND tv.objective = 0 "
            "ORDER BY t.number"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print(f"[!] No completed trials found in {db_path}.")
            return

        # Simple Optimization History Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="number", y="value", marker="o")
        plt.title(f"Optimization History: {model_label.upper()}")
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value (Lower is Better)")
        plt.grid(True, linestyle="--", alpha=0.6)

        plot_path = os.path.join(
            output_dir, f"optuna_{model_label}_history.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"[✓] Optuna plot saved to: {plot_path}")

    @staticmethod
    def visualize_attention(
        model_id: str,
        prompt: str,
        tokenizer_path: str,
        output_path: str = "report/plots/attention_map.png",
    ):
        """
        Generates a heatmap of self-attention weights for a given prompt.
        Uses the standard Transformers library.
        """
        print(f"[+] Generating attention map for {model_id}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                output_attentions=True,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True,
            )
            model.config.output_attentions = True

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

            if outputs.attentions is None or not outputs.attentions:
                print(
                    "[!] Attention tensors are not available for this model "
                    "configuration. Skipping attention map."
                )
                return

            # Extract attentions
            # (tuple of layers, each [batch, heads, seq, seq])
            # We take the last layer and the average of all heads
            # We cast to float32
            # because numpy/matplotlib
            # don't support bfloat16
            attention = (
                outputs.attentions[-1][0]
                .mean(dim=0)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="viridis",
            )
            plt.title(f"Self-Attention Weights (Last Layer) - {model_id}")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            print(f"[✓] Attention map saved to: {output_path}")

        except Exception as e:
            print(f"[!] Failed to generate attention map: {e}")


def run_visualizations():
    """CLI Entry point to generate all project visualizations."""
    print("\n=== NLP Transformers: Generating Visualizations ===")
    viz = Visualizer()

    # 1. Optuna Plots (Check results/optuna directory)
    viz.plot_optuna_results("results/optuna/optuna_qwen.db", "qwen")
    viz.plot_optuna_results("results/optuna/optuna_llama.db", "llama")

    # 2. Attention Visualization (Sample)
    sample_prompt = "Reserve a table for 2 at PF Changs tomorrow at 6pm."

    from app.models.config import model_settings

    # Qwen Attention Map
    viz.visualize_attention(
        model_id="Qwen/Qwen3.5-2B",
        prompt=sample_prompt,
        tokenizer_path=str(model_settings.qwen_tok_path),
        output_path="report/plots/attention_qwen.png",
    )

    # Llama Attention Map
    viz.visualize_attention(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        prompt=sample_prompt,
        tokenizer_path=str(model_settings.llama_tok_path),
        output_path="report/plots/attention_llama.png",
    )


if __name__ == "__main__":
    run_visualizations()
