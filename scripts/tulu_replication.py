"""Tulu entry point for the source-backed original execution protocol."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('baseline', 'train'):
        raise SystemExit('The reconstructed baseline/train interface is retired. Use --config, --country, --output; '
                         'historical diagnostics are in tulu_reconstructed.py with explicit opt-in.')
    if '--family' in sys.argv:
        raise SystemExit('This entry point fixes family=tulu. Use original_protocol.py for other families.')
    from scripts.original_protocol import main
    main(default_family='tulu')
