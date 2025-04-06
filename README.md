# Financial Analysis MCP Server

A powerful server for financial analysis capabilities, providing stock data retrieval and visualization features.

## Features

- Stock data retrieval from yfinance
- Alpha Vantage API integration
- Web search for financial information
- Interactive visualization with Plotly
- RESTful API endpoints

## Prerequisites

- Python 3.8+
- Alpha Vantage API key (for additional data)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fin-mcp
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Alpha Vantage API key:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Usage

1. Start the server:
```bash
cd src
python main.py
```

2. The server will start at `http://localhost:8000`

3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

### Stock Data
- `GET /api/v1/stock/{symbol}` - Get stock data from yfinance
- `GET /api/v1/alpha-vantage/{symbol}` - Get stock data from Alpha Vantage
- `GET /api/v1/search/{query}` - Search for financial information

### Visualization
- `GET /api/v1/visualize/candlestick/{symbol}` - Get candlestick chart
- `GET /api/v1/visualize/volume/{symbol}` - Get volume chart
- `GET /api/v1/visualize/combined/{symbol}` - Get combined price and volume chart

## Example Usage

```python
import requests

# Get stock data
response = requests.get("http://localhost:8000/api/v1/stock/AAPL")
data = response.json()

# Get candlestick chart
response = requests.get("http://localhost:8000/api/v1/visualize/candlestick/AAPL")
chart_data = response.json()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
