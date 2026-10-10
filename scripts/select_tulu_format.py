"""Choose a pilot input format transparently from the baseline diagnostics."""
import argparse
import json
import math
from pathlib import Path


def select_format(report):
    formats = report['formats']
    for name in ('legacy-pair', 'chat-template'):
        entry = formats[name]
        if len(entry['countries']) != 7 or not math.isfinite(entry['mean_absolute_difference_pp']):
            raise ValueError('Incomplete baseline report')
    matching = [name for name, entry in formats.items() if entry['maximum_absolute_error_pp'] <= .05]
    chosen = min(matching, key=lambda name: formats[name]['mean_absolute_difference_pp']) if matching else 'chat-template'
    return dict(input_format=chosen, baseline_matches_paper=bool(matching),
                reason='all countries within 0.05 percentage points of published global baseline'
                if matching else 'no verified baseline match; use official chat template for explicitly provisional pilot',
                baseline_errors={name: dict(mean_absolute_difference_pp=v['mean_absolute_difference_pp'],
                                           maximum_absolute_error_pp=v['maximum_absolute_error_pp'])
                                 for name, v in formats.items()},
                historical_training_recipe_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select_format(json.loads(args.report.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['input_format'])
