import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any
import json

class VisualizationService:
    def create_candlestick_chart(self, data: Dict[str, Any], title: str = "Stock Price") -> Dict[str, Any]:
        """Create an interactive candlestick chart"""
        df = pd.DataFrame(data)
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark'
        )
        
        return json.loads(fig.to_json())

    def create_volume_chart(self, data: Dict[str, Any], title: str = "Trading Volume") -> Dict[str, Any]:
        """Create an interactive volume chart"""
        df = pd.DataFrame(data)
        
        fig = go.Figure(data=[go.Bar(
            x=df.index,
            y=df['Volume']
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_dark'
        )
        
        return json.loads(fig.to_json())

    def create_combined_chart(self, price_data: Dict[str, Any], volume_data: Dict[str, Any], 
                            title: str = "Stock Analysis") -> Dict[str, Any]:
        """Create a combined price and volume chart"""
        price_df = pd.DataFrame(price_data)
        volume_df = pd.DataFrame(volume_data)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=price_df.index,
            open=price_df['Open'],
            high=price_df['High'],
            low=price_df['Low'],
            close=price_df['Close']
        ), row=1, col=1)
        
        # Add volume chart
        fig.add_trace(go.Bar(
            x=volume_df.index,
            y=volume_df['Volume']
        ), row=2, col=1)
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            template='plotly_dark'
        )
        
        return json.loads(fig.to_json()) 