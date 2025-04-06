from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from src.services.financial_data import FinancialDataService
from src.services.visualization import VisualizationService

router = APIRouter()
financial_service = FinancialDataService()
visualization_service = VisualizationService()

@router.get("/stock/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y"):
    """Get stock data from yfinance"""
    return await financial_service.get_stock_data(symbol, period)

@router.get("/alpha-vantage/{symbol}")
async def get_alpha_vantage_data(symbol: str, function: str = "TIME_SERIES_INTRADAY", interval: str = "5min", outputsize: str = "compact"):
    """Get stock data from Alpha Vantage
    
    Args:
        symbol (str): The stock symbol to fetch data for
        function (str): The Alpha Vantage API function to call
        interval (str): Time interval between two consecutive data points
        outputsize (str): Size of the output data (compact or full)
    """
    return await financial_service.get_alpha_vantage_data(symbol, function, interval, outputsize)

@router.get("/search/{query}")
async def search_financial_data(query: str):
    """Search for financial information on the web"""
    return await financial_service.search_web_data(query)

@router.get("/visualize/candlestick/{symbol}")
async def get_candlestick_chart(symbol: str, period: str = "1y"):
    """Get candlestick chart for a stock"""
    data = await financial_service.get_stock_data(symbol, period)
    return visualization_service.create_candlestick_chart(
        data["historical_data"],
        title=f"{symbol} Stock Price"
    )

@router.get("/visualize/volume/{symbol}")
async def get_volume_chart(symbol: str, period: str = "1y"):
    """Get volume chart for a stock"""
    data = await financial_service.get_stock_data(symbol, period)
    return visualization_service.create_volume_chart(
        data["historical_data"],
        title=f"{symbol} Trading Volume"
    )

@router.get("/visualize/combined/{symbol}")
async def get_combined_chart(symbol: str, period: str = "1y"):
    """Get combined price and volume chart for a stock"""
    data = await financial_service.get_stock_data(symbol, period)
    return visualization_service.create_combined_chart(
        data["historical_data"],
        data["historical_data"],
        title=f"{symbol} Stock Analysis"
    ) 