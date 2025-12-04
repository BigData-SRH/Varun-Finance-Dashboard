# 📈 Finance Dashboard

An interactive Streamlit dashboard for analyzing NSE (National Stock Exchange of India) indices and their constituent stocks. Built with Python, powered by data from Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📑 Table of Contents

- [Key Performance Indicators](#-key-performance-indicators-kpis)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Indices Covered](#-indices-covered)
- [Dashboard Pages](#️-dashboard-pages)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## 📊 Key Performance Indicators (KPIs)

| Category | KPIs |
|----------|------|
| **Returns** | Total Return, CAGR, Win Rate |
| **Risk** | Annualized Volatility, Maximum Drawdown, Beta |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Fundamentals** | PE Ratio, PB Ratio, Dividend Yield, Market Cap |

---

## 🚀 Getting Started

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

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
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

## 📈 Indices Covered

| Index | Ticker | Stocks | Description |
|-------|--------|--------|-------------|
| **Nifty 50** | ^NSEI | 50 | Top 50 companies by market cap |
| **Nifty Bank** | ^NSEBANK | 12 | Major banking sector stocks |
| **Nifty IT** | ^CNXIT | 10 | Leading IT sector companies |

**Analysis Period:** January 2015 – Present

---

## 🖥️ Dashboard Pages

### Home
- Real-time index cards with price and daily change
- Normalized performance comparison with **time period filters** (1W, 1M, 6M, 1Y, 5Y, MAX)
- Correlation heatmap between indices
- Detailed analysis for selected index (CAGR, Volatility, Sharpe, Drawdown)
- **Global Market Context**: Compare Indian indices against S&P 500, NASDAQ, FTSE 100, DAX, Nikkei 225, and Hang Seng
- **Educational Section**: Learn what each metric means and how to invest in indices via ETFs and Index Funds

### Stock Explorer
- Filter by index and sector
- Individual stock KPIs: CAGR, Volatility, Beta, PE Ratio, Dividend Yield
- **Corporate Events** tab: Dividend history, stock splits, upcoming earnings
- Risk-return scatter visualization

### Portfolio
- Top performers by returns, Sharpe ratio, and dividends
- Value opportunities (Low PE with strong CAGR)
- AI-suggested diversified portfolio based on composite scoring

### About
- Methodology documentation
- Data quality scoring system
- Full list of index constituents

---

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **Data Source:** Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance)
- **Visualization:** Plotly
- **Analysis:** Pandas, NumPy

---

## 📁 Project Structure

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

## ⚙️ Configuration

Edit `utils/config.py` to customize:

```python
START_DATE = '2015-01-01'      # Analysis start date
RISK_FREE_RATE = 0.06          # 6% annual rate
CONSTITUENTS = {...}           # Index member stocks
```

---

## ⚠️ Disclaimer

This dashboard is for **educational and informational purposes only**. It does not constitute financial advice. Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.

Market data is sourced from Yahoo Finance via the [yfinance](https://github.com/ranaroussi/yfinance) library. This project is not affiliated with or endorsed by Yahoo.

---

## 📜 License

MIT License © 2025
