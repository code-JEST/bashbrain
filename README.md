# 🧠 BashBrain

A tiny AI trained from scratch that autocompletes bash commands.
Type the beginning of a command and BashBrain will suggest how to finish it.

## Demo

![BashBrain demo](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNzhnamZ0OXI5enQ0NGptYnppbzdka3ViemI0MW5mcXN0bmw5ZTV4byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pqB0H0V5NqenQSuGwb/giphy.gif)

## Educational project
This project is intentionally built on microgpt.py — a dependency-free GPT written in pure Python with no NumPy or PyTorch. This makes it extremely slow to train compared to real ML frameworks, which means the model has only seen a fraction of the data it would need to become truly useful.

The goal was never to build a production tool.

## How it works

BashBrain is built on [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy — a fully self-contained GPT implementation in pure Python with zero dependencies. It includes:

- A hand-written autograd engine for backpropagation
- A transformer architecture with multi-head attention, RMSNorm, and MLP blocks
- An Adam optimizer
- Character-level tokenization

The model was trained on ~11,000 real bash commands from the [NL2Bash dataset](https://github.com/TellinaTool/nl2bash), learning bash syntax and common command patterns entirely from scratch.

## Features

- 🔡 Character-level GPT trained on real bash commands
- ⚡ Autocompletes partial commands like `find . -name` or `grep -r`
- 🖥️ Simple web UI built with Streamlit
- 🧪 No ML frameworks — pure Python only

## Project structure

```
bashbrain/
├── microgpt.py        # GPT model + autograd engine (by Karpathy)
├── prepare_data.py    # Downloads and cleans the bash dataset
├── train.py           # Trains the model and saves weights
├── app.py             # Streamlit web app
```

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/code-JEST/bashbrain.git
cd bashbrain
```

**2. Install dependencies**
```bash
pip install streamlit
```

**3. Prepare the dataset and train the model**
```bash
python prepare_data.py
python train.py
```
Training takes around 5–15 minutes depending on your machine.

**4. Run the app**
```bash
streamlit run app.py
```

## Example output

```
Input:   find . -name
Output:  find . -name "*.txt" -type f

Input:   grep -r
Output:  grep -r "pattern" /path/to/dir

Input:   tar -
Output:  tar -czf archive.tar.gz folder/
```

## What I learned

Through this project I gained practical experience with:

- Understanding how backpropagation works by building parts of the training loop and autograd logic myself
- Collecting, cleaning and structuring text data for language model training
- Deploying a small machine learning application as a web interface

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for microgpt.py
- [TellinaTool](https://github.com/TellinaTool/nl2bash) for the NL2Bash dataset
