# Finance Dashboard

An interactive Streamlit dashboard for analyzing NSE (National Stock Exchange of India) indices and their constituent stocks. Built with Python, powered by data from Yahoo Finance.

**Author:** Varun H S | **LinkedIn:** [linkedin.com/in/varunhs306](https://www.linkedin.com/in/varunhs306)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Key Performance Indicators](#key-performance-indicators-kpis)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Indices Covered](#indices-covered)
- [Dashboard Pages](#dashboard-pages)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Key Performance Indicators (KPIs)

| Category | KPIs |
|----------|------|
| **Returns** | Total Return, CAGR, Win Rate |
| **Risk** | Annualized Volatility, Maximum Drawdown, Beta |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Fundamentals** | PE Ratio, PB Ratio, Dividend Yield, Market Cap |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/BigData-SRH/Varun-Finance-Dashboard.git
cd Varun-Finance-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run Home.py
```

The dashboard will open at `http://localhost:8501`

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and runnn
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the container
docker-compose down
```

### Using Docker Directly

```bash
# Build the image
docker build -t varun-finance-dashboard .

# Run the container
docker run -p 8501:8501 varun-finance-dashboard
```

The dashboard will be available at `http://localhost:8501`

---

## Indices Covered

| Index | Ticker | Stocks | Description |
|-------|--------|--------|-------------|
| **Nifty 50** | ^NSEI | 50 | Top 50 companies by market cap |
| **Nifty Bank** | ^NSEBANK | 12 | Major banking sector stocks |
| **Nifty IT** | ^CNXIT | 10 | Leading IT sector companies |

**Analysis Period:** January 2015 – Present

---

## Dashboard Pages

### Home
Real-time index tracking, KPI summary cards (Performance/Risk/Technical), normalized performance charts with time filters, side-by-side index comparison, correlation analysis, global market benchmarking, and educational resources.

### Stock Explorer
Card-based metrics layout, technical indicators with visual badges, price history charts, performance analysis, corporate events tracking, and risk-return visualization.

### Portfolio
Top performers analysis, value opportunities identification, and AI-suggested diversified portfolio recommendations.

### About
Methodology documentation, data quality scoring, and complete index constituents listing.

---

## Tech Stack

- **Framework:** Streamlit
- **Data Source:** Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance)
- **Visualization:** Plotly
- **Analysis:** Pandas, NumPy

---

## Features

- **Interactive Cards**: Hover effects and visual depth with enhanced shadows
- **Real-time Data**: Live market data from Yahoo Finance
- **Technical Analysis**: RSI, Moving Averages, 52-week highs/lows
- **Global Comparison**: Benchmark Indian indices against global markets
- **Responsive Design**: Clean, modern UI with organized layouts

---

## Project Structure

```
├── Home.py                 # Main entry point
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose orchestration
├── .dockerignore           # Docker build exclusions
├── requirements.txt        # Python dependencies
├── pages/
│   ├── 02_Stock_Explorer.py
│   ├── 03_Portfolio.py
│   └── 04_About.py
├── utils/
│   ├── config.py           # Configuration (dates, indices, colors)
│   ├── data_fetcher.py     # Yahoo Finance API wrapper
│   ├── metrics.py          # Financial calculations
│   ├── quality.py          # Data quality scoring
│   └── charts.py           # Plotly chart functions
└── README.md
```

---

## Configuration

Edit `utils/config.py` to customize:

```python
START_DATE = '2015-01-01'      # Analysis start date
RISK_FREE_RATE = 0.06          # 6% annual rate
CONSTITUENTS = {...}           # Index member stocks
```

---

## Disclaimer

This dashboard is for **educational and informational purposes only**. It does not constitute financial advice. Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.

Market data is sourced from Yahoo Finance via the [yfinance](https://github.com/ranaroussi/yfinance) library. This project is not affiliated with or endorsed by Yahoo.

---

## License

MIT License © 2025 Varun H S | LinkedIn: [linkedin.com/in/varunhs306](https://www.linkedin.com/in/varunhs306)
