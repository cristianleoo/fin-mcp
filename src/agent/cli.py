import asyncio
import argparse
from typing import Optional
from .financial_agent import FinancialAgent
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

async def analyze_stock(symbol: str, period: str = "1y", output_format: str = "table"):
    """Analyze a stock and display results"""
    agent = FinancialAgent()
    
    try:
        # Get analysis
        analysis = await agent.analyze_stock(symbol, period)
        insights = await agent.generate_insights(symbol, period)
        
        if output_format == "json":
            console.print(json.dumps(analysis, indent=2))
            console.print("\nInsights:")
            console.print(json.dumps(insights, indent=2))
        else:
            # Display analysis in table format
            table = Table(title=f"Stock Analysis for {symbol}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Current Price", f"${analysis['current_price']:.2f}")
            table.add_row("Price Change", f"{analysis['price_change_percent']:.2f}%")
            table.add_row("Average Volume", f"{analysis['average_volume']:,.0f}")
            table.add_row("Volatility", f"{analysis['volatility']:.2f}%")
            
            # Add moving averages
            ma_table = Table(title="Moving Averages")
            ma_table.add_column("Period", style="cyan")
            ma_table.add_column("Value", style="magenta")
            for period, value in analysis['moving_averages'].items():
                ma_table.add_row(period, f"${value:.2f}")
            
            # Add insights
            insights_table = Table(title="Insights")
            insights_table.add_column("Category", style="cyan")
            insights_table.add_column("Value", style="magenta")
            
            insights_table.add_row("Trend", insights['trend'])
            insights_table.add_row("Support", f"${insights['support_resistance']['support']:.2f}")
            insights_table.add_row("Resistance", f"${insights['support_resistance']['resistance']:.2f}")
            insights_table.add_row("Risk Level", insights['risk_assessment']['risk_level'])
            insights_table.add_row("Recommendation", insights['llm_analysis']['recommendation'])
            
            console.print(table)
            console.print("\n")
            console.print(ma_table)
            console.print("\n")
            console.print(insights_table)
            
            # Display reasoning
            console.print("\n")
            # console.print(Panel(insights['recommendation']['reasoning'], title="Recommendation Reasoning"))
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Financial Analysis Agent CLI")
    parser.add_argument("symbol", help="Stock symbol to analyze")
    parser.add_argument("--period", default="1y", help="Time period for analysis (default: 1y)")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    
    args = parser.parse_args()
    
    asyncio.run(analyze_stock(args.symbol, args.period, args.format))

if __name__ == "__main__":
    main() 