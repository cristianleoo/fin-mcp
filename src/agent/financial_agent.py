import requests
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import logging
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialAgent:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.openai_key = os.getenv("GEMINI_API_KEY")
        self.client = OpenAI(api_key=self.openai_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.session = requests.Session()
        logger.info(f"Initialized FinancialAgent with base_url: {base_url}")

    async def get_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get stock data directly from yfinance"""
        logger.info(f"Fetching stock data for {symbol} with period {period}")
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            logger.info(f"Successfully retrieved historical data for {symbol}")
            
            # Convert historical data to dictionary format
            historical_data = {
                "Date": hist.index.strftime('%Y-%m-%d').tolist(),
                "Open": hist["Open"].tolist(),
                "High": hist["High"].tolist(),
                "Low": hist["Low"].tolist(),
                "Close": hist["Close"].tolist(),
                "Volume": hist["Volume"].tolist(),
                "Dividends": hist["Dividends"].tolist(),
                "Stock Splits": hist["Stock Splits"].tolist()
            }
            
            # Get basic info
            info = {
                "symbol": symbol,
                "name": stock.info.get('longName', ''),
                "sector": stock.info.get('sector', ''),
                "industry": stock.info.get('industry', ''),
                "marketCap": stock.info.get('marketCap', 0),
                "currency": stock.info.get('currency', 'USD'),
                "dividendYield": stock.info.get('dividendYield', 0),
                "trailingPE": stock.info.get('trailingPE', 0),
                "forwardPE": stock.info.get('forwardPE', 0),
                "beta": stock.info.get('beta', 0),
                "52WeekHigh": stock.info.get('fiftyTwoWeekHigh', 0),
                "52WeekLow": stock.info.get('fiftyTwoWeekLow', 0)
            }
            logger.info(f"Retrieved basic info for {symbol}")
            
            # Get additional data
            additional_data = {}
            
            # Handle recommendations
            try:
                rec = stock.recommendations
                additional_data["recommendations"] = rec.to_dict() if rec is not None else None
                logger.info(f"Retrieved recommendations for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get recommendations for {symbol}: {str(e)}")
                additional_data["recommendations"] = None

            # Handle financial statements
            try:
                financials = stock.financials
                additional_data["financials"] = financials.to_dict() if financials is not None else None
                logger.info(f"Retrieved financials for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get financials for {symbol}: {str(e)}")
                additional_data["financials"] = None

            try:
                quarterly_financials = stock.quarterly_financials
                additional_data["quarterly_financials"] = quarterly_financials.to_dict() if quarterly_financials is not None else None
                logger.info(f"Retrieved quarterly financials for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get quarterly financials for {symbol}: {str(e)}")
                additional_data["quarterly_financials"] = None

            # Handle balance sheet
            try:
                balance_sheet = stock.balance_sheet
                additional_data["balance_sheet"] = balance_sheet.to_dict() if balance_sheet is not None else None
                logger.info(f"Retrieved balance sheet for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get balance sheet for {symbol}: {str(e)}")
                additional_data["balance_sheet"] = None

            try:
                quarterly_balance_sheet = stock.quarterly_balance_sheet
                additional_data["quarterly_balance_sheet"] = quarterly_balance_sheet.to_dict() if quarterly_balance_sheet is not None else None
                logger.info(f"Retrieved quarterly balance sheet for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get quarterly balance sheet for {symbol}: {str(e)}")
                additional_data["quarterly_balance_sheet"] = None

            # Handle cash flow
            try:
                cashflow = stock.cashflow
                additional_data["cashflow"] = cashflow.to_dict() if cashflow is not None else None
                logger.info(f"Retrieved cash flow for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get cash flow for {symbol}: {str(e)}")
                additional_data["cashflow"] = None

            try:
                quarterly_cashflow = stock.quarterly_cashflow
                additional_data["quarterly_cashflow"] = quarterly_cashflow.to_dict() if quarterly_cashflow is not None else None
                logger.info(f"Retrieved quarterly cash flow for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get quarterly cash flow for {symbol}: {str(e)}")
                additional_data["quarterly_cashflow"] = None

            # Handle earnings
            try:
                earnings = stock.earnings
                additional_data["earnings"] = earnings.to_dict() if earnings is not None else None
                logger.info(f"Retrieved earnings for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get earnings for {symbol}: {str(e)}")
                additional_data["earnings"] = None

            try:
                quarterly_earnings = stock.quarterly_earnings
                additional_data["quarterly_earnings"] = quarterly_earnings.to_dict() if quarterly_earnings is not None else None
                logger.info(f"Retrieved quarterly earnings for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get quarterly earnings for {symbol}: {str(e)}")
                additional_data["quarterly_earnings"] = None

            # Handle other data
            try:
                earnings_dates = stock.earnings_dates
                additional_data["earnings_dates"] = earnings_dates.to_dict() if earnings_dates is not None else None
                logger.info(f"Retrieved earnings dates for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get earnings dates for {symbol}: {str(e)}")
                additional_data["earnings_dates"] = None

            try:
                institutional_holders = stock.institutional_holders
                additional_data["institutional_holders"] = institutional_holders.to_dict() if institutional_holders is not None else None
                logger.info(f"Retrieved institutional holders for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get institutional holders for {symbol}: {str(e)}")
                additional_data["institutional_holders"] = None

            try:
                major_holders = stock.major_holders
                additional_data["major_holders"] = major_holders.to_dict() if major_holders is not None else None
                logger.info(f"Retrieved major holders for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get major holders for {symbol}: {str(e)}")
                additional_data["major_holders"] = None

            try:
                sustainability = stock.sustainability
                additional_data["sustainability"] = sustainability.to_dict() if sustainability is not None else None
                logger.info(f"Retrieved sustainability data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get sustainability data for {symbol}: {str(e)}")
                additional_data["sustainability"] = None

            try:
                additional_data["news"] = stock.news
                logger.info(f"Retrieved news for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get news for {symbol}: {str(e)}")
                additional_data["news"] = None
            
            return {
                "historical_data": historical_data,
                "info": info,
                "additional_data": additional_data
            }
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            raise Exception(f"Error fetching stock data: {str(e)}")

    async def get_alpha_vantage_data(self, symbol: str, interval: str = "5min", function: str = "TIME_SERIES_INTRADAY") -> Dict[str, Any]:
        """Get stock data from Alpha Vantage API
        
        Args:
            symbol (str): The stock symbol to fetch data for
            interval (str): Time interval between two consecutive data points. 
                           Valid values: 1min, 5min, 15min, 30min, 60min
            function (str): The Alpha Vantage API function to call. Default is TIME_SERIES_INTRADAY.
                           Other options include:
                           - TIME_SERIES_DAILY
                           - TIME_SERIES_WEEKLY
                           - TIME_SERIES_MONTHLY
                           - GLOBAL_QUOTE
                           - OVERVIEW
                           - EARNINGS
                           - INCOME_STATEMENT
                           - BALANCE_SHEET
                           - CASH_FLOW
        """
        try:
            logger.info(f"Fetching Alpha Vantage data for {symbol} with function {function}")
            
            # Construct the API endpoint
            url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": interval if function == "TIME_SERIES_INTRADAY" else None,
                "outputsize": "compact",  # Default to compact for faster response
                "apikey": self.av_api_key
            }
            
            # Remove None values from params
            params = {k: v for k, v in params.items() if v is not None}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Process the data based on the function type
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f"Time Series ({interval})"
                if time_series_key in data:
                    # Convert the time series data to a more structured format
                    processed_data = {
                        "metadata": data.get("Meta Data", {}),
                        "time_series": [
                            {
                                "timestamp": timestamp,
                                "open": float(values["1. open"]),
                                "high": float(values["2. high"]),
                                "low": float(values["3. low"]),
                                "close": float(values["4. close"]),
                                "volume": int(values["5. volume"])
                            }
                            for timestamp, values in data[time_series_key].items()
                        ]
                    }
                    return processed_data
            elif function == "GLOBAL_QUOTE":
                if "Global Quote" in data:
                    return {
                        "symbol": data["Global Quote"]["01. symbol"],
                        "open": float(data["Global Quote"]["02. open"]),
                        "high": float(data["Global Quote"]["03. high"]),
                        "low": float(data["Global Quote"]["04. low"]),
                        "price": float(data["Global Quote"]["05. price"]),
                        "volume": int(data["Global Quote"]["06. volume"]),
                        "latest_trading_day": data["Global Quote"]["07. latest trading day"],
                        "previous_close": float(data["Global Quote"]["08. previous close"]),
                        "change": float(data["Global Quote"]["09. change"]),
                        "change_percent": data["Global Quote"]["10. change percent"]
                    }
            elif function in ["OVERVIEW", "EARNINGS", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]:
                return data
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            raise Exception(f"Error fetching Alpha Vantage data: {str(e)}")

    async def search_financial_info(self, query: str) -> Dict[str, Any]:
        """Search for financial information"""
        response = self.session.get(f"{self.base_url}/search/{query}")
        response.raise_for_status()
        return response.json()

    async def analyze_stock(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Perform comprehensive stock analysis"""
        logger.info(f"Starting analysis for {symbol}")
        try:
            # Get basic stock data
            stock_data = await self.get_stock_data(symbol, period)
            logger.info(f"Retrieved stock data for {symbol}")
            
            # Convert historical data to DataFrame
            historical_data = pd.DataFrame(stock_data["historical_data"])
            historical_data["Date"] = pd.to_datetime(historical_data["Date"])
            historical_data.set_index("Date", inplace=True)
            
            info = stock_data["info"]
            additional_data = stock_data["additional_data"]
            
            # Calculate technical indicators
            analysis = {
                "symbol": symbol,
                "current_price": historical_data["Close"].iloc[-1],
                "price_change": historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[0],
                "price_change_percent": ((historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[0]) / historical_data["Close"].iloc[0]) * 100,
                "average_volume": historical_data["Volume"].mean(),
                "volatility": historical_data["Close"].pct_change().std() * 100,
                "moving_averages": {
                    "20_day": historical_data["Close"].rolling(window=20).mean().iloc[-1],
                    "50_day": historical_data["Close"].rolling(window=50).mean().iloc[-1],
                    "200_day": historical_data["Close"].rolling(window=200).mean().iloc[-1]
                },
                "company_info": info,
                "valuation_metrics": {
                    "pe_ratio": info.get("trailingPE", 0),
                    "forward_pe": info.get("forwardPE", 0),
                    "dividend_yield": info.get("dividendYield", 0),
                    "beta": info.get("beta", 0),
                    "market_cap": info.get("marketCap", 0)
                },
                "price_ranges": {
                    "52_week_high": info.get("52WeekHigh", 0),
                    "52_week_low": info.get("52WeekLow", 0),
                    "current_vs_high": ((info.get("52WeekHigh", 0) - historical_data["Close"].iloc[-1]) / info.get("52WeekHigh", 0)) * 100 if info.get("52WeekHigh", 0) != 0 else 0,
                    "current_vs_low": ((historical_data["Close"].iloc[-1] - info.get("52WeekLow", 0)) / info.get("52WeekLow", 0)) * 100 if info.get("52WeekLow", 0) != 0 else 0
                },
                "historical_data": historical_data
            }
            logger.info(f"Calculated technical indicators for {symbol}")

            # Add financial analysis if available
            if additional_data["financials"] is not None:
                try:
                    financials = pd.DataFrame(additional_data["financials"])
                    analysis["financial_metrics"] = {
                        "revenue_growth": self._calculate_growth_rate(financials, "Total Revenue"),
                        "profit_margin": self._calculate_profit_margin(financials),
                        "debt_to_equity": self._calculate_debt_to_equity(pd.DataFrame(additional_data["balance_sheet"])) if additional_data["balance_sheet"] is not None else 0
                    }
                    logger.info(f"Calculated financial metrics for {symbol}")
                except Exception as e:
                    logger.warning(f"Error processing financials for {symbol}: {str(e)}")
                    analysis["financial_metrics"] = None

            # Add earnings analysis if available
            if additional_data["earnings"] is not None:
                try:
                    earnings = pd.DataFrame(additional_data["earnings"])
                    analysis["earnings_analysis"] = {
                        "earnings_growth": self._calculate_growth_rate(earnings, "Earnings"),
                        "eps_trend": self._analyze_eps_trend(earnings)
                    }
                    logger.info(f"Calculated earnings analysis for {symbol}")
                except Exception as e:
                    logger.warning(f"Error processing earnings for {symbol}: {str(e)}")
                    analysis["earnings_analysis"] = None

            # Add institutional ownership analysis if available
            if additional_data["institutional_holders"] is not None:
                try:
                    inst_holders = pd.DataFrame(additional_data["institutional_holders"])
                    analysis["ownership"] = {
                        "institutional_ownership": inst_holders["Shares"].sum() if "Shares" in inst_holders.columns else 0,
                        "major_holders": additional_data["major_holders"]
                    }
                    logger.info(f"Calculated ownership analysis for {symbol}")
                except Exception as e:
                    logger.warning(f"Error processing institutional holders for {symbol}: {str(e)}")
                    analysis["ownership"] = None

            # Add recommendations analysis if available
            if additional_data["recommendations"] is not None:
                try:
                    recommendations = pd.DataFrame(additional_data["recommendations"])
                    if not recommendations.empty:
                        analysis["recommendations"] = {
                            "current_rating": self._get_current_recommendation(recommendations),
                            "rating_trend": self._analyze_recommendation_trend(recommendations)
                        }
                        logger.info(f"Calculated recommendations analysis for {symbol}")
                    else:
                        logger.warning(f"No recommendations data available for {symbol}")
                        analysis["recommendations"] = None
                except Exception as e:
                    logger.warning(f"Error processing recommendations for {symbol}: {str(e)}")
                    analysis["recommendations"] = None

            # Get additional data from Alpha Vantage if available
            try:
                alpha_data = await self.get_alpha_vantage_data(symbol)
                analysis["alpha_vantage_data"] = alpha_data
                logger.info(f"Retrieved Alpha Vantage data for {symbol}")
            except Exception as e:
                logger.warning(f"Error getting Alpha Vantage data for {symbol}: {str(e)}")
                analysis["alpha_vantage_data"] = None

            # Get recent news and information
            try:
                news = await self.search_financial_info(f"{symbol} stock news")
                analysis["recent_news"] = news
                logger.info(f"Retrieved news for {symbol}")
            except Exception as e:
                logger.warning(f"Error getting news for {symbol}: {str(e)}")
                analysis["recent_news"] = None

            logger.info(f"Completed analysis for {symbol}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            raise

    def _calculate_growth_rate(self, df: pd.DataFrame, column: str) -> float:
        """Calculate growth rate for a given column"""
        if column not in df.columns:
            return 0
        values = df[column].dropna()
        if len(values) < 2:
            return 0
        return ((values.iloc[0] - values.iloc[-1]) / values.iloc[-1]) * 100

    def _calculate_profit_margin(self, financials: pd.DataFrame) -> float:
        """Calculate profit margin"""
        if "Net Income" not in financials.columns or "Total Revenue" not in financials.columns:
            return 0
        return (financials["Net Income"].iloc[0] / financials["Total Revenue"].iloc[0]) * 100

    def _calculate_debt_to_equity(self, balance_sheet: pd.DataFrame) -> float:
        """Calculate debt to equity ratio"""
        if balance_sheet is None:
            return 0
        if "Total Debt" not in balance_sheet.columns or "Total Stockholder Equity" not in balance_sheet.columns:
            return 0
        return balance_sheet["Total Debt"].iloc[0] / balance_sheet["Total Stockholder Equity"].iloc[0]

    def _analyze_eps_trend(self, earnings: pd.DataFrame) -> Dict[str, Any]:
        """Analyze EPS trend"""
        if "Earnings" not in earnings.columns:
            return {"trend": "Unknown", "growth_rate": 0}
        
        eps_values = earnings["Earnings"].dropna()
        if len(eps_values) < 2:
            return {"trend": "Insufficient Data", "growth_rate": 0}
        
        growth_rate = ((eps_values.iloc[0] - eps_values.iloc[-1]) / eps_values.iloc[-1]) * 100
        trend = "Growing" if growth_rate > 0 else "Declining"
        
        return {
            "trend": trend,
            "growth_rate": growth_rate
        }

    def _get_current_recommendation(self, recommendations: pd.DataFrame) -> Dict[str, Any]:
        """Get current recommendation"""
        if recommendations is None or recommendations.empty:
            logger.warning("No recommendations data available")
            return {"rating": "Unknown", "date": None}
        
        try:
            latest = recommendations.iloc[0]
            return {
                "rating": latest.get("To Grade", "Unknown"),
                "date": latest.get("Date", None)
            }
        except Exception as e:
            logger.error(f"Error getting current recommendation: {str(e)}")
            return {"rating": "Unknown", "date": None}

    def _analyze_recommendation_trend(self, recommendations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recommendation trend"""
        if recommendations is None or recommendations.empty:
            logger.warning("No recommendations data available for trend analysis")
            return {"trend": "Unknown", "changes": 0}
        
        try:
            # Count upgrades and downgrades
            if "Action" in recommendations.columns:
                changes = recommendations["Action"].value_counts().to_dict()
                return {
                    "trend": "Positive" if changes.get("up", 0) > changes.get("down", 0) else "Negative",
                    "changes": changes
                }
            else:
                logger.warning("No 'Action' column in recommendations data")
                return {"trend": "Unknown", "changes": 0}
        except Exception as e:
            logger.error(f"Error analyzing recommendation trend: {str(e)}")
            return {"trend": "Unknown", "changes": 0}

    async def generate_insights(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Generate insights and recommendations based on analysis using LLM"""
        analysis = await self.analyze_stock(symbol, period)
        historical_data = pd.DataFrame(analysis["historical_data"])
        
        # Generate LLM-based insights
        llm_insights = await self._generate_llm_insights(analysis)
        
        # Generate technical recommendation
        technical_recommendation = self._generate_technical_recommendation(analysis, historical_data)
        
        # Combine technical and LLM-based insights
        insights = {
            "symbol": symbol,
            "trend": self._determine_trend(historical_data),
            "support_resistance": self._calculate_support_resistance(historical_data),
            "risk_assessment": self._assess_risk(historical_data),
            "technical_recommendation": technical_recommendation,
            "llm_analysis": llm_insights,
            "company_info": analysis["company_info"]
        }

        return insights

    def _generate_technical_recommendation(self, analysis: Dict[str, Any], historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate investment recommendation based on technical analysis"""
        current_price = analysis["current_price"]
        price_change = analysis["price_change_percent"]
        trend = self._determine_trend(historical_data)
        
        # Calculate moving average crossovers
        ma_20 = historical_data["Close"].rolling(window=20).mean()
        ma_50 = historical_data["Close"].rolling(window=50).mean()
        ma_200 = historical_data["Close"].rolling(window=200).mean()
        
        # Check for golden cross (20-day MA crosses above 50-day MA)
        golden_cross = ma_20.iloc[-1] > ma_50.iloc[-1] and ma_20.iloc[-2] <= ma_50.iloc[-2]
        # Check for death cross (20-day MA crosses below 50-day MA)
        death_cross = ma_20.iloc[-1] < ma_50.iloc[-1] and ma_20.iloc[-2] >= ma_50.iloc[-2]
        
        # Determine recommendation based on multiple factors
        if trend == "Bullish" and price_change > 0 and golden_cross:
            recommendation = "Strong Buy"
            confidence = "High"
            reasoning = "Bullish trend with positive price momentum and golden cross formation"
        elif trend == "Bearish" and price_change < 0 and death_cross:
            recommendation = "Strong Sell"
            confidence = "High"
            reasoning = "Bearish trend with negative price momentum and death cross formation"
        elif trend == "Bullish" and price_change > 0:
            recommendation = "Buy"
            confidence = "Medium"
            reasoning = "Bullish trend with positive price momentum"
        elif trend == "Bearish" and price_change < 0:
            recommendation = "Sell"
            confidence = "Medium"
            reasoning = "Bearish trend with negative price momentum"
        else:
            recommendation = "Hold"
            confidence = "Medium"
            reasoning = "Neutral trend with mixed signals"

        return {
            "action": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "technical_indicators": {
                "trend": trend,
                "price_change": f"{price_change:.2f}%",
                "ma_20": ma_20.iloc[-1],
                "ma_50": ma_50.iloc[-1],
                "ma_200": ma_200.iloc[-1],
                "golden_cross": golden_cross,
                "death_cross": death_cross
            }
        }

    async def _generate_llm_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights using OpenAI's LLM"""
        try:
            # Prepare the prompt with relevant data
            prompt = f"""
            Analyze the following stock data and provide insights:

            Company: {analysis['company_info'].get('name', 'Unknown')} ({analysis['symbol']})
            Sector: {analysis['company_info'].get('sector', 'Unknown')}
            Current Price: ${analysis['current_price']:.2f}
            Price Change: {analysis['price_change_percent']:.2f}%
            Market Cap: ${analysis['valuation_metrics']['market_cap']:,.2f}
            P/E Ratio: {analysis['valuation_metrics']['pe_ratio']:.2f}
            Beta: {analysis['valuation_metrics']['beta']:.2f}
            Dividend Yield: {analysis['valuation_metrics']['dividend_yield']:.2f}%
            
            Technical Analysis:
            - 20-day MA: ${analysis['moving_averages']['20_day']:.2f}
            - 50-day MA: ${analysis['moving_averages']['50_day']:.2f}
            - 200-day MA: ${analysis['moving_averages']['200_day']:.2f}
            - Volatility: {analysis['volatility']:.2f}%
            
            Please provide your analysis in the following format:
            
            Overview:
            [Your overview here]
            
            Strengths and Weaknesses:
            [Your analysis of strengths and weaknesses]
            
            Short-term Outlook:
            [Your short-term analysis]
            
            Long-term Outlook:
            [Your long-term analysis]
            
            Recommendation:
            [Your recommendation and reasoning]
            
            Risks:
            [Your risk assessment]
            """

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing detailed stock analysis and investment recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            # Get the response text
            llm_response = response.choices[0].message.content

            # Return the structured insights
            return {
                "overview": self._extract_section(llm_response, "Overview"),
                "strengths_weaknesses": self._extract_section(llm_response, "Strengths and Weaknesses"),
                "outlook": {
                    "short_term": self._extract_section(llm_response, "Short-term Outlook"),
                    "long_term": self._extract_section(llm_response, "Long-term Outlook")
                },
                "recommendation": self._extract_section(llm_response, "Recommendation"),
                "risks": self._extract_section(llm_response, "Risks"),
                "raw_analysis": llm_response
            }

        except Exception as e:
            logger.error(f"Error generating LLM insights: {str(e)}")
            return {
                "error": "Failed to generate LLM insights",
                "details": str(e)
            }

    def _extract_section(self, text: str, section: str) -> str:
        """Extract a specific section from the LLM response"""
        try:
            # This is a simple extraction method - you might want to improve it
            # based on how the LLM structures its response
            lines = text.split('\n')
            section_lines = []
            in_section = False
            
            for line in lines:
                if section.lower() in line.lower():
                    in_section = True
                    continue
                if in_section and line.strip() and not any(other_section in line.lower() for other_section in ['overview', 'strengths', 'weaknesses', 'outlook', 'recommendation', 'risks']):
                    section_lines.append(line)
                elif in_section and line.strip():
                    break
            
            return '\n'.join(section_lines).strip()
        except Exception as e:
            logger.warning(f"Error extracting section {section}: {str(e)}")
            return "Section extraction failed"

    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine the current trend of the stock"""
        short_ma = data["Close"].rolling(window=20).mean()
        long_ma = data["Close"].rolling(window=50).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return "Bullish"
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return "Bearish"
        else:
            return "Neutral"

    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        recent_data = data.tail(30)
        return {
            "support": recent_data["Low"].min(),
            "resistance": recent_data["High"].max()
        }

    def _assess_risk(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the risk level of the stock"""
        volatility = data["Close"].pct_change().std() * 100
        if volatility < 10:
            risk_level = "Low"
        elif volatility < 20:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "risk_level": risk_level,
            "volatility": volatility
        } 