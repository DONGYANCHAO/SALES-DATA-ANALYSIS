"""
Sales Data Analysis Module

This module provides functionality for analyzing online sales data including:
- Loading and cleaning sales data from CSV files
- Analyzing top products and customers by total sales
- Generating visualization charts (bar charts and line charts)

Author: AI Assistant
Date: 2026-04-14
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SalesDataError(Exception):
    """Base exception for sales data analysis errors."""
    pass


class FileNotFoundError(SalesDataError):
    """Raised when the data file is not found."""
    pass


class DataFormatError(SalesDataError):
    """Raised when data format is invalid."""
    pass


class EmptyDataError(SalesDataError):
    """Raised when data is empty after cleaning."""
    pass


@dataclass
class Config:
    """Configuration class for sales data analysis.
    
    Attributes:
        data_path: Relative path to the CSV data file
        output_dir: Relative path to the output directory for visualizations
        chart_figsize: Tuple of (width, height) for chart figure size
        chart_color_product_bar: Color for product bar chart
        chart_color_product_line: Color for product line chart
        chart_color_customer_bar: Color for customer bar chart
        columns_to_clean: List of column names to clean (drop NaN values)
        top_n: Number of top items to analyze
    """
    data_path: Path = field(default_factory=lambda: Path("./DATASET/online_sales_dataset.csv"))
    output_dir: Path = field(default_factory=lambda: Path("./visuals/"))
    chart_figsize: tuple = (10, 6)
    chart_color_product_bar: str = "teal"
    chart_color_product_line: str = "blue"
    chart_linestyle: str = "--"
    chart_marker: str = "o"
    chart_markersize: int = 10
    chart_color_customer_bar: str = "coral"
    columns_to_clean: list = field(default_factory=lambda: ["CustomerID", "ShippingCost", "WarehouseLocation"])
    top_n: int = 10
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Constants
INVOICE_DATE_COLUMN: str = "InvoiceDate"
QUANTITY_COLUMN: str = "Quantity"
UNIT_PRICE_COLUMN: str = "UnitPrice"
DISCOUNT_COLUMN: str = "Discount"
TOTAL_SALES_COLUMN: str = "Total Sales"
DESCRIPTION_COLUMN: str = "Description"
CUSTOMER_ID_COLUMN: str = "CustomerID"


def load_data(config: Config) -> DataFrame:
    """Load sales data from CSV file.
    
    Args:
        config: Configuration object containing data path
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the data file does not exist
        DataFormatError: If the file cannot be parsed as CSV
    """
    try:
        logger.info(f"Loading data from {config.data_path}")
        
        if not config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {config.data_path}")
        
        df = pd.read_csv(config.data_path)
        logger.info(f"Successfully loaded {len(df)} rows from {config.data_path}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data file is empty: {e}")
        raise DataFormatError(f"Data file is empty: {e}")
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise DataFormatError(f"Failed to parse CSV: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise SalesDataError(f"Unexpected error loading data: {e}")


def clean_data(df: DataFrame, config: Config) -> DataFrame:
    """Clean the sales data by removing rows with missing values.
    
    Args:
        df: Input DataFrame
        config: Configuration object containing columns to clean
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        EmptyDataError: If no data remains after cleaning
        DataFormatError: If required columns are missing
    """
    try:
        logger.info("Starting data cleaning")
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        logger.debug(f"Original data shape: {cleaned_df.shape}")
        logger.debug(f"Null values before cleaning:\n{cleaned_df.isna().sum()}")
        
        # Check if required columns exist
        missing_columns = [col for col in config.columns_to_clean if col not in cleaned_df.columns]
        if missing_columns:
            raise DataFormatError(f"Missing required columns: {missing_columns}")
        
        # Drop rows with NaN values in specified columns
        cleaned_df = cleaned_df.dropna(subset=config.columns_to_clean)
        logger.debug(f"Null values after cleaning:\n{cleaned_df.isna().sum()}")
        
        # Convert InvoiceDate to datetime
        if INVOICE_DATE_COLUMN in cleaned_df.columns:
            cleaned_df[INVOICE_DATE_COLUMN] = pd.to_datetime(
                cleaned_df[INVOICE_DATE_COLUMN], 
                errors='coerce'
            )
            logger.debug(f"Converted {INVOICE_DATE_COLUMN} to datetime")
        
        # Check if data is empty after cleaning
        if cleaned_df.empty:
            raise EmptyDataError("No data remains after cleaning")
        
        logger.info(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
        return cleaned_df
        
    except EmptyDataError:
        raise
    except DataFormatError:
        raise
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise SalesDataError(f"Error during data cleaning: {e}")


def calculate_total_sales(df: DataFrame) -> DataFrame:
    """Calculate Total Sales column based on Quantity, UnitPrice, and Discount.
    
    Formula: Total Sales = Quantity * UnitPrice * (1 - Discount)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with Total Sales column added
        
    Raises:
        DataFormatError: If required columns are missing
    """
    try:
        required_columns = [QUANTITY_COLUMN, UNIT_PRICE_COLUMN, DISCOUNT_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataFormatError(f"Missing required columns for total sales calculation: {missing_columns}")
        
        result_df = df.copy()
        result_df[TOTAL_SALES_COLUMN] = (
            result_df[QUANTITY_COLUMN] * 
            result_df[UNIT_PRICE_COLUMN] * 
            (1 - result_df[DISCOUNT_COLUMN])
        )
        
        logger.debug(f"Calculated {TOTAL_SALES_COLUMN} column")
        return result_df
        
    except DataFormatError:
        raise
    except Exception as e:
        logger.error(f"Error calculating total sales: {e}")
        raise SalesDataError(f"Error calculating total sales: {e}")


def analyze_top_products(df: DataFrame, config: Config) -> Series:
    """Analyze top N products by total sales.
    
    Args:
        df: Input DataFrame with Total Sales column
        config: Configuration object containing top_n
        
    Returns:
        Series containing top N products by total sales
        
    Raises:
        DataFormatError: If required columns are missing
    """
    try:
        if DESCRIPTION_COLUMN not in df.columns:
            raise DataFormatError(f"Missing required column: {DESCRIPTION_COLUMN}")
        if TOTAL_SALES_COLUMN not in df.columns:
            raise DataFormatError(f"Missing required column: {TOTAL_SALES_COLUMN}")
        
        logger.info(f"Analyzing top {config.top_n} products by total sales")
        
        top_products = (
            df.groupby(DESCRIPTION_COLUMN)[TOTAL_SALES_COLUMN]
            .sum()
            .sort_values(ascending=False)
            .head(config.top_n)
        )
        
        logger.info(f"Top {config.top_n} products analysis complete")
        logger.debug(f"Top products:\n{top_products}")
        
        return top_products
        
    except DataFormatError:
        raise
    except Exception as e:
        logger.error(f"Error analyzing top products: {e}")
        raise SalesDataError(f"Error analyzing top products: {e}")


def analyze_top_customers(df: DataFrame, config: Config) -> Series:
    """Analyze top N customers by purchase volume (total sales).
    
    Args:
        df: Input DataFrame with Total Sales column
        config: Configuration object containing top_n
        
    Returns:
        Series containing top N customers by total sales
        
    Raises:
        DataFormatError: If required columns are missing
    """
    try:
        if CUSTOMER_ID_COLUMN not in df.columns:
            raise DataFormatError(f"Missing required column: {CUSTOMER_ID_COLUMN}")
        if TOTAL_SALES_COLUMN not in df.columns:
            raise DataFormatError(f"Missing required column: {TOTAL_SALES_COLUMN}")
        
        logger.info(f"Analyzing top {config.top_n} customers by purchase volume")
        
        # Convert CustomerID to Int64 for proper display
        analysis_df = df.copy()
        analysis_df[CUSTOMER_ID_COLUMN] = analysis_df[CUSTOMER_ID_COLUMN].astype('Int64')
        
        top_customers = (
            analysis_df.groupby(CUSTOMER_ID_COLUMN)[TOTAL_SALES_COLUMN]
            .sum()
            .sort_values(ascending=False)
            .head(config.top_n)
        )
        
        logger.info(f"Top {config.top_n} customers analysis complete")
        logger.debug(f"Top customers:\n{top_customers}")
        
        return top_customers
        
    except DataFormatError:
        raise
    except Exception as e:
        logger.error(f"Error analyzing top customers: {e}")
        raise SalesDataError(f"Error analyzing top customers: {e}")


def create_bar_chart(
    data: Series,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    config: Config
) -> plt.Figure:
    """Create a bar chart visualization.
    
    Args:
        data: Series containing data to plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        config: Configuration object
        
    Returns:
        Matplotlib Figure object
    """
    try:
        logger.debug(f"Creating bar chart: {title}")
        
        fig, ax = plt.subplots(figsize=config.chart_figsize)
        data.plot(kind='bar', color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        logger.debug(f"Bar chart created: {title}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")
        raise SalesDataError(f"Error creating bar chart: {e}")


def create_line_chart(
    data: Series,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    config: Config
) -> plt.Figure:
    """Create a line chart visualization.
    
    Args:
        data: Series containing data to plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
        config: Configuration object
        
    Returns:
        Matplotlib Figure object
    """
    try:
        logger.debug(f"Creating line chart: {title}")
        
        fig, ax = plt.subplots(figsize=config.chart_figsize)
        data.plot(
            kind='line',
            color=color,
            linestyle=config.chart_linestyle,
            marker=config.chart_marker,
            markersize=config.chart_markersize,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
        # Set x-tick labels to product names
        labels = data.index.tolist()
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        
        logger.debug(f"Line chart created: {title}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating line chart: {e}")
        raise SalesDataError(f"Error creating line chart: {e}")


def save_chart(fig: plt.Figure, filename: str, config: Config) -> None:
    """Save chart to file and close figure to prevent memory leaks.
    
    Args:
        fig: Matplotlib Figure object
        filename: Output filename
        config: Configuration object containing output directory
    """
    try:
        output_path = config.output_dir / filename
        logger.info(f"Saving chart to {output_path}")
        
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Chart saved successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving chart: {e}")
        plt.close(fig)
        raise SalesDataError(f"Error saving chart: {e}")


def main(config: Optional[Config] = None) -> None:
    """Main function to execute sales data analysis pipeline.
    
    Args:
        config: Optional configuration object. Uses default Config if not provided.
    """
    if config is None:
        config = Config()
    
    try:
        logger.info("Starting sales data analysis")
        
        # Load data
        df = load_data(config)
        logger.info(f"Data loaded: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        logger.debug(f"Info:\n{df.info()}")
        logger.debug(f"Statistical information:\n{df.describe()}")
        
        # Clean data
        df_cleaned = clean_data(df, config)
        
        # Calculate Total Sales
        df_with_sales = calculate_total_sales(df_cleaned)
        
        # Analyze top products
        top_products = analyze_top_products(df_with_sales, config)
        logger.info(f"TOP {config.top_n} PRODUCTS")
        logger.info(f"\n{top_products}")
        
        # Create and save bar chart for top products
        fig_bar = create_bar_chart(
            top_products,
            title="TOP 10 PRODUCTS BY SALE",
            xlabel="product description",
            ylabel="Total Sales",
            color=config.chart_color_product_bar,
            config=config
        )
        save_chart(fig_bar, "top_products_bar.png", config)
        
        # Create and save line chart for top products
        fig_line = create_line_chart(
            top_products,
            title="TOP 10 PRODUCTS BY SALE",
            xlabel="product description",
            ylabel="Total Sales",
            color=config.chart_color_product_line,
            config=config
        )
        save_chart(fig_line, "top_products_line.png", config)
        
        # Analyze top customers
        top_customers = analyze_top_customers(df_with_sales, config)
        logger.info(f"TOP {config.top_n} CUSTOMERS")
        logger.info(f"\n{top_customers}")
        
        # Create and save bar chart for top customers
        fig_cust = create_bar_chart(
            top_customers,
            title="TOP 10 CUSTOMERS BY PURCHASE VOLUME",
            xlabel="CUSTOMER ID",
            ylabel="TOTAL SALES",
            color=config.chart_color_customer_bar,
            config=config
        )
        save_chart(fig_cust, "top_customers_bar.png", config)
        
        logger.info("Sales data analysis completed successfully")
        
    except SalesDataError as e:
        logger.error(f"Sales data analysis failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise SalesDataError(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
