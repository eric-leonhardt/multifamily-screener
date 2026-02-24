# Multifamily Investment Screener

AI-powered tool for identifying undervalued multifamily rental properties. Upload your data, chat naturally, and get a full LASSO regression analysis with acquisition targets — no coding required.

**[→ Try the live app](https://multifamily-screener-ai.streamlit.app/)**

<p align="center">
  <img src="assets/map_example.png" alt="Submarket map with acquisition targets" width="700">
  <br>
  <em>Properties colored by model residual — red = under-rented acquisition targets</em>
</p>

## How It Works

The app uses Claude (Anthropic's LLM) as a conversational layer on top of a LASSO regression pipeline. Upload a CSV of property-level rent data (Yardi Matrix, CoStar, or similar), and Claude walks you through the analysis:

1. **Inspects your data** — identifies columns, submarkets, and feature types automatically
2. **Guides feature selection** — suggests which variables to include based on what's in your dataset
3. **Engineers distance features** — computes Haversine distances to landmarks you specify
4. **Runs cross-validated LASSO** — automatic variable selection with L1 regularization
5. **Ranks variable importance** — partial R² measures each feature's unique contribution
6. **Identifies acquisition targets** — properties renting significantly below model-predicted values
7. **Explains everything in plain English** — what the model found, why it matters, and what to look at

## Quick Start

### AI Chat App (recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste your [Anthropic API key](https://console.anthropic.com/) in the sidebar, upload a CSV, and start chatting.

### CLI (no API key needed)

```bash
python run_analysis.py
```

The CLI walks you through the same analysis interactively in the terminal. Save your config as YAML to rerun without prompts:

```bash
python run_analysis.py --config my_config.yaml
```

## Architecture

```
User (chat) → Streamlit → Claude API (tool-use)
                               ↓
                          tools.py (pipeline functions)
                               ↓
                          Claude interprets & explains
                               ↓
                          User (plain English + map)
```

Claude has access to four tools:

| Tool | Purpose |
|------|---------|
| `inspect_data` | Understand dataset structure |
| `list_submarkets` | Show available submarkets with property counts |
| `describe_columns` | Auto-classify features (binary, numeric, categorical) |
| `run_analysis` | Full LASSO pipeline — model, importance, targets, map |

## Project Structure

```
├── app.py              # Streamlit chat interface (AI version)
├── tools.py            # Pipeline functions as Claude tool calls
├── run_analysis.py     # Standalone CLI version
├── utils.py            # Haversine, encoding, imputation helpers
├── requirements.txt
└── assets/
    └── map_example.png
```

## Requirements

- Python 3.8+
- Anthropic API key (for the chat app)
- See `requirements.txt` for packages

## License

MIT
