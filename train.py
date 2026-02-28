import subprocess
import sys

subprocess.run([
    sys.executable, "microgpt.py",
    "--n_embd", "64",
    "--n_layer", "4",
    "--n_head", "4",
    "--block_size", "32",
    "--num_steps", "5000",
    "--learning_rate", "1e-2"
])