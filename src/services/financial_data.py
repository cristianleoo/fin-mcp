import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
import os
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

class FinancialDataService:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')

    async def get_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            # Get basic info without using the problematic info property
            info = {
                "symbol": symbol,
                "name": stock.info.get('longName', ''),
                "sector": stock.info.get('sector', ''),
                "industry": stock.info.get('industry', ''),
                "marketCap": stock.info.get('marketCap', 0),
                "currency": stock.info.get('currency', 'USD')
            }
            
            return {
                "historical_data": hist.to_dict(),
                "info": info
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching stock data: {str(e)}")

    async def get_alpha_vantage_data(self, symbol: str, function: str = "TIME_SERIES_INTRADAY", interval: str = "5min", outputsize: str = "compact") -> Dict[str, Any]:
        """Get stock data from Alpha Vantage
        
        Args:
            symbol (str): The stock symbol to fetch data for
            function (str): The Alpha Vantage API function to call
            interval (str): Time interval between two consecutive data points
            outputsize (str): Size of the output data (compact or full)
        """
        try:
            if not self.alpha_vantage_key:
                raise HTTPException(status_code=400, detail="Alpha Vantage API key not configured")
            
            if function == "TIME_SERIES_INTRADAY":
                data, _ = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
                return {"intraday_data": data.to_dict()}
            elif function == "TIME_SERIES_DAILY":
                data, _ = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
                return {"daily_data": data.to_dict()}
            elif function == "TIME_SERIES_WEEKLY":
                data, _ = self.ts.get_weekly(symbol=symbol)
                return {"weekly_data": data.to_dict()}
            elif function == "TIME_SERIES_MONTHLY":
                data, _ = self.ts.get_monthly(symbol=symbol)
                return {"monthly_data": data.to_dict()}
            elif function == "GLOBAL_QUOTE":
                data, _ = self.ts.get_quote_endpoint(symbol=symbol)
                return {"quote": data.to_dict()}
            elif function == "OVERVIEW":
                data, _ = self.fd.get_company_overview(symbol=symbol)
                return {"overview": data.to_dict()}
            elif function == "EARNINGS":
                data, _ = self.fd.get_earnings(symbol=symbol)
                return {"earnings": data.to_dict()}
            elif function == "INCOME_STATEMENT":
                data, _ = self.fd.get_income_statement_annual(symbol=symbol)
                return {"income_statement": data.to_dict()}
            elif function == "BALANCE_SHEET":
                data, _ = self.fd.get_balance_sheet_annual(symbol=symbol)
                return {"balance_sheet": data.to_dict()}
            elif function == "CASH_FLOW":
                data, _ = self.fd.get_cash_flow_annual(symbol=symbol)
                return {"cash_flow": data.to_dict()}
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported function: {function}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching Alpha Vantage data: {str(e)}")

    async def search_web_data(self, query: str) -> Dict[str, Any]:
        """Search for financial information on the web"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            search_url = f"https://www.google.com/search?q={query}+financial+news"
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant information
            results = []
            for result in soup.find_all('div', class_='g'):
                title = result.find('h3')
                if title:
                    results.append({
                        'title': title.text,
                        'link': result.find('a')['href'] if result.find('a') else None
                    })
            
            return {"search_results": results}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error searching web data: {str(e)}") 