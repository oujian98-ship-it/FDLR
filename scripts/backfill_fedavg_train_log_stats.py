#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backfill final FedAvg stats into timestamped train logs."""

import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_log(log_path):
    text = log_path.read_text(encoding='utf-8')
    model_match = re.search(r'^model_name:\s*(.+)$', text, re.MULTILINE)
    result_match = re.search(r'^result_folder:\s*(.+)$', text, re.MULTILINE)
    if not model_match or not result_match:
        return None, None, text
    return model_match.group(1).strip(), Path(result_match.group(1).strip()), text


def read_result_csv(result_folder, model_name):
    csv_path = result_folder / f'{model_name}.csv'
    if not csv_path.exists():
        return None
    losses = []
    params = []
    runtime = None
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == 'loss':
                continue
            if row[0] == 'runtime':
                runtime = float(row[1])
            else:
                losses.append(float(row[0]))
                params.append(int(float(row[1])))
    if not params:
        return None
    return {
        'final_avg_loss': f'{losses[-1]:.12f}' if losses else '',
        'final_round_params_shared': params[-1],
        'cumulative_params_shared': sum(params),
        'final_round_N_x1e6_params': f'{params[-1] / 1_000_000:.6f}',
        'communication_N_x1e6_params': f'{sum(params) / 1_000_000:.6f}',
        'csv_runtime_sec': f'{runtime:.6f}' if runtime is not None else '',
    }


def backfill(log_path):
    model_name, result_folder, text = parse_log(log_path)
    if not model_name or not result_folder:
        return False
    stats = read_result_csv(result_folder, model_name)
    if not stats:
        return False

    block = ['\n[Final stats]\n']
    for key, value in stats.items():
        block.append(f'{key}: {value}\n')

    marker = '\n[Artifacts]\n'
    if '[Final stats]' in text:
        text = re.sub(
            r'\n\[Final stats\]\n.*?(?=\n\[Artifacts\]|\Z)',
            ''.join(block).rstrip(),
            text,
            flags=re.S,
        )
    elif marker in text:
        text = text.replace(marker, ''.join(block) + marker, 1)
    else:
        text = text.rstrip() + ''.join(block) + '\n'
    log_path.write_text(text, encoding='utf-8')
    return True


def main():
    changed = 0
    train_root = PROJECT_ROOT / 'logs' / 'train_logs'
    for log_path in train_root.glob('fedavg_*/*.log'):
        if backfill(log_path):
            changed += 1
            print(f'Updated {log_path}')
    print(f'Backfilled {changed} FedAvg train logs.')


if __name__ == '__main__':
    main()
