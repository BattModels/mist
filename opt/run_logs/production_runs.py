import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich import print as rprint

def get_production_pretraining(log_dir="/projects/bcuf/abhutani/electrolyte-fm/.cache/wandb-export"):
    
    production_models = {
        "MIST-1.8B": "dh61satt",
        "MIST-28M": "ti624ev1"
    }
    
    pretraining_logs = Path(log_dir) / "pretraining"
    results = {}
    
    for model_name, run_id in production_models.items():
        json_path = pretraining_logs / f"{run_id}.json"
        if json_path.exists():
            results[model_name] = json.loads(json_path.read_text())
        else:
            rprint(f"Logs Not Found for {model_name}: {json_path}")
            results[model_name] = None
    
    print_summary(results)
    return results

def maybe_impute_step(data):
    step =  data.get('trainer', {}).get('step', None)
    if step is None:
        step = max(data.get('metric_traces').get('step'))
    return step

def maybe_impute_tokens(data):
    toks =  data.get('trainer', {}).get('total_tokens_step', None)
    if toks is None:
        toks = data.get('metric_traces').get('total_tokens_step', None)
        if toks in not None:
            return max(toks)
    return toks

def print_summary(results):
    console = Console()
        
    # Comparison table
    table = Table(title="Model Comparison", show_header=True)
    table.add_column("Metric")
    
    for model_name, data in results.items():
        if data is not None:
            table.add_column(model_name, justify="right")
    
    # Extract metrics
    metrics = [
        ("Run ID", lambda d: d.get('id', 'N/A')),
        ("State", lambda d: d.get('state', 'N/A')),
        ("Parameters", lambda d: f"{d.get('model', {}).get('model_size', 0)/1e6:.1f}M"),
        ("Hidden Size", lambda d: f"{d.get('model', {}).get('d_model', 0):,}"),
        ("Layers", lambda d: str(d.get('model', {}).get('n_layers', 'N/A'))),
        ("Steps", lambda d: f"{maybe_impute_step(d):,}"),
        ("Tokens", lambda d: f"{maybe_impute_tokens(d)/1e9:.2f}B"),
        ("Val Loss (best)", lambda d: f"{d.get('metrics', {}).get('val_loss_best', float('nan')):.4f}"),
        ("Throughput", lambda d: f"{d.get('system', {}).get('train_throughput', 0):.2f} b/s"),
        ("Runtime", lambda d: f"{d.get('runtime', 0)/3600:.2f}h"),
        ("Cluster", lambda d: d.get('cluster', 'N/A')),
    ]
    
    for metric_name, extractor in metrics:
        row = [metric_name]
        for model_name, data in results.items():
            print(data, extractor)
            if data:
                row.append(extractor(data))
            else:
                row.append("N/A")
        table.add_row(*row)
    
    console.print(table)
    
    # Full JSON for each model
    for model_name, data in results.items():
        if data:
            console.print(f"{model_name} - Full Details")
            console.print(Panel(JSON(json.dumps(data, indent=2)), title=f"{model_name} Raw Data", expand=False))

if __name__ == "__main__":
    results = get_production_pretraining()
