import sys
from pathlib import Path
from interact import plot_attentions_over_time

if len(sys.argv) != 2:
    print("Usage: python gen_attention_viz.py <path to model dir>")
    sys.exit(1)

dip = sys.argv[1]
dip = Path(f"models/{dip}")
plot_attentions_over_time(dip)
