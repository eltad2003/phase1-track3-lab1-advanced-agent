from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)


def _run_with_progress(agent, examples, label: str) -> list:
    total = len(examples)
    records = []
    for idx, example in enumerate(examples, start=1):
        records.append(agent.run(example))
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"[yellow]{label}[/yellow]: {idx}/{total}")
    return records


@app.command()
def main(dataset: str = "data/hotpot_100.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3) -> None:
    examples = load_dataset(dataset)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"[cyan]Dataset[/cyan]: {dataset} ({len(examples)} questions)")
    print(f"[cyan]Runtime[/cyan]: openai / {model}")
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    print("[bold]Running ReAct...[/bold]")
    react_records = _run_with_progress(react, examples, "ReAct")
    print("[bold]Running Reflexion...[/bold]")
    reflexion_records = _run_with_progress(reflexion, examples, "Reflexion")
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(
        dataset).name, mode="openai")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
