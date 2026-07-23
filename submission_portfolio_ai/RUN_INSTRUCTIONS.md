# Run Instructions

## 1) Create environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure environment variables
```bash
cp .env.example .env
```
Fill required keys in `.env`:
- `ANTHROPIC_API_KEY`
- `ROBINHOOD_USERNAME` / `ROBINHOOD_PASSWORD` (optional if using imported portfolio)
- `FMP_API_KEY` (optional)
- `GEMINI_API_KEY` (optional, screenshot importer backend)

## 3) Run pipeline
Full pipeline:
```bash
python main.py
```

Data/quant smoke test without committee or PDF:
```bash
python main.py --skip-committee --skip-pdf
```

## 4) Run dashboard
```bash
streamlit run reporting/dashboard.py
```

## 5) Run import page
```bash
streamlit run reporting/pages/import_portfolio.py
```

## 6) Optional: use included sample artifacts
Included sample outputs:
- `latest_run.json`
- `consistency_audit.json`
- `citation_audit.json`
- report drafts (`research_paper_detailed.md`, `project_report_draft.md`)

