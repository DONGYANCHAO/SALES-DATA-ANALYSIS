"""
销售数据分析模块

该模块提供销售数据的加载、清洗、分析和可视化功能。
主要功能包括：
- 从CSV文件加载销售数据
- 数据清洗和预处理
- 分析热门产品和客户
- 生成柱状图和折线图
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


class DataLoadError(Exception):
    """数据加载异常"""
    pass


class DataCleanError(Exception):
    """数据清洗异常"""
    pass


class AnalysisError(Exception):
    """数据分析异常"""
    pass


class ChartError(Exception):
    """图表生成异常"""
    pass


@dataclass
class Config:
    """全局配置类"""
    DATA_PATH: Path = field(default_factory=lambda: Path('./DATASET/online_sales_dataset.csv'))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path('./visuals'))
    CLEAN_COLUMNS: List[str] = field(default_factory=lambda: ['CustomerID', 'ShippingCost', 'WarehouseLocation'])
    FIGURE_SIZE: Tuple[int, int] = (10, 6)
    BAR_COLOR_PRODUCTS: str = 'teal'
    BAR_COLOR_CUSTOMERS: str = 'coral'
    LINE_COLOR: str = 'blue'
    LINE_STYLE: str = '--'
    MARKER: str = 'o'
    MARKER_SIZE: int = 10
    TOP_N: int = 10

    def __post_init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path: Path) -> pd.DataFrame:
    """
    加载CSV数据文件

    Args:
        file_path: 数据文件路径

    Returns:
        加载的DataFrame

    Raises:
        DataLoadError: 文件不存在或数据格式错误时抛出
    """
    try:
        if not file_path.exists():
            raise DataLoadError(f"数据文件不存在: {file_path}")
        
        LOGGER.info(f"正在加载数据文件: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise DataLoadError("数据文件为空")
        
        LOGGER.info(f"数据加载成功，共 {len(df)} 条记录")
        LOGGER.info(f"数据列: {df.columns.tolist()}")
        LOGGER.debug(f"数据列详情: {df.columns}")
        LOGGER.info(f"数据信息:\n{df.info()}")
        LOGGER.info(f"统计信息:\n{df.describe()}")
        LOGGER.info(f"空值统计:\n{df.isna().sum()}")
        
        return df
    except pd.errors.EmptyDataError:
        raise DataLoadError("数据文件为空或格式错误")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"数据解析错误: {e}")
    except Exception as e:
        if isinstance(e, DataLoadError):
            raise
        raise DataLoadError(f"加载数据时发生错误: {e}")


def clean_data(df: pd.DataFrame, clean_columns: List[str]) -> pd.DataFrame:
    """
    清洗数据

    Args:
        df: 原始DataFrame
        clean_columns: 需要清洗空值的列名列表

    Returns:
        清洗后的DataFrame

    Raises:
        DataCleanError: 数据清洗过程中发生错误时抛出
    """
    try:
        if df is None or df.empty:
            raise DataCleanError("输入数据为空")
        
        LOGGER.info("开始数据清洗")
        df_cleaned = df.copy()
        
        initial_count = len(df_cleaned)
        df_cleaned.dropna(subset=clean_columns, inplace=True)
        dropped_count = initial_count - len(df_cleaned)
        
        LOGGER.info(f"删除了 {dropped_count} 条包含空值的记录")
        LOGGER.info(f"清洗后空值统计:\n{df_cleaned.isna().sum()}")
        
        df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'], errors='coerce')
        
        df_cleaned['Total Sales'] = (
            df_cleaned['Quantity'] * 
            df_cleaned['UnitPrice'] * 
            (1 - df_cleaned['Discount'])
        )
        LOGGER.info("已计算 Total Sales 列")
        
        df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype('Int64')
        LOGGER.info("已转换 CustomerID 为 Int64 类型")
        
        return df_cleaned
    except KeyError as e:
        raise DataCleanError(f"缺少必要的列: {e}")
    except Exception as e:
        if isinstance(e, DataCleanError):
            raise
        raise DataCleanError(f"数据清洗时发生错误: {e}")


def analyze_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    分析销量最高的产品

    Args:
        df: 清洗后的DataFrame
        top_n: 返回前N个产品

    Returns:
        销量最高的N个产品的Series

    Raises:
        AnalysisError: 分析过程中发生错误时抛出
    """
    try:
        if df is None or df.empty:
            raise AnalysisError("输入数据为空")
        
        if 'Description' not in df.columns or 'Total Sales' not in df.columns:
            raise AnalysisError("缺少必要的列: Description 或 Total Sales")
        
        LOGGER.info(f"分析销量最高的 {top_n} 个产品")
        top_products = (
            df.groupby('Description')['Total Sales']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        
        LOGGER.info(f"销量最高的 {top_n} 个产品:\n{top_products}")
        
        return top_products
    except Exception as e:
        if isinstance(e, AnalysisError):
            raise
        raise AnalysisError(f"分析产品时发生错误: {e}")


def analyze_top_customers(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    分析购买量最高的客户

    Args:
        df: 清洗后的DataFrame
        top_n: 返回前N个客户

    Returns:
        购买量最高的N个客户的Series

    Raises:
        AnalysisError: 分析过程中发生错误时抛出
    """
    try:
        if df is None or df.empty:
            raise AnalysisError("输入数据为空")
        
        if 'CustomerID' not in df.columns or 'Total Sales' not in df.columns:
            raise AnalysisError("缺少必要的列: CustomerID 或 Total Sales")
        
        LOGGER.info(f"分析购买量最高的 {top_n} 个客户")
        top_customers = (
            df.groupby('CustomerID')['Total Sales']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        
        LOGGER.info(f"购买量最高的 {top_n} 个客户:\n{top_customers}")
        
        return top_customers
    except Exception as e:
        if isinstance(e, AnalysisError):
            raise
        raise AnalysisError(f"分析客户时发生错误: {e}")


def save_chart(fig: plt.Figure, output_path: Path) -> None:
    """
    保存图表到文件

    Args:
        fig: matplotlib图表对象
        output_path: 输出文件路径

    Raises:
        ChartError: 保存图表时发生错误时抛出
    """
    try:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        LOGGER.info(f"图表已保存: {output_path}")
    except Exception as e:
        raise ChartError(f"保存图表时发生错误: {e}")
    finally:
        plt.close(fig)


def create_bar_chart(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    output_filename: str,
    config: Config
) -> Optional[plt.Figure]:
    """
    创建柱状图

    Args:
        data: 要可视化的数据
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        color: 柱状图颜色
        output_filename: 输出文件名
        config: 配置对象

    Returns:
        matplotlib图表对象，如果出错返回None

    Raises:
        ChartError: 创建图表时发生错误时抛出
    """
    try:
        if data is None or data.empty:
            raise ChartError("输入数据为空")
        
        LOGGER.info(f"创建柱状图: {title}")
        
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        data.plot(kind='bar', color=color, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = config.OUTPUT_DIR / output_filename
        save_chart(fig, output_path)
        
        return fig
    except Exception as e:
        if isinstance(e, ChartError):
            raise
        raise ChartError(f"创建柱状图时发生错误: {e}")


def create_line_chart(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    line_style: str,
    marker: str,
    marker_size: int,
    output_filename: str,
    config: Config
) -> Optional[plt.Figure]:
    """
    创建折线图

    Args:
        data: 要可视化的数据
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        color: 线条颜色
        line_style: 线条样式
        marker: 标记样式
        marker_size: 标记大小
        output_filename: 输出文件名
        config: 配置对象

    Returns:
        matplotlib图表对象，如果出错返回None

    Raises:
        ChartError: 创建图表时发生错误时抛出
    """
    try:
        if data is None or data.empty:
            raise ChartError("输入数据为空")
        
        LOGGER.info(f"创建折线图: {title}")
        
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        data.plot(kind='line', color=color, linestyle=line_style, 
                  marker=marker, markersize=marker_size, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        xtick_labels = data.index.tolist()
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90)
        
        plt.tight_layout()
        plt.grid(True)
        
        output_path = config.OUTPUT_DIR / output_filename
        save_chart(fig, output_path)
        
        return fig
    except Exception as e:
        if isinstance(e, ChartError):
            raise
        raise ChartError(f"创建折线图时发生错误: {e}")


def main() -> None:
    """
    主函数，执行完整的销售数据分析流程
    """
    LOGGER.info("=" * 50)
    LOGGER.info("开始销售数据分析")
    LOGGER.info("=" * 50)
    
    config = Config()
    
    try:
        df = load_data(config.DATA_PATH)
        
        df_cleaned = clean_data(df, config.CLEAN_COLUMNS)
        
        top_products = analyze_top_products(df_cleaned, config.TOP_N)
        
        top_customers = analyze_top_customers(df_cleaned, config.TOP_N)
        
        create_bar_chart(
            data=top_products,
            title="TOP 10 PRODUCTS BY SALE",
            xlabel="product description",
            ylabel="Total Sales",
            color=config.BAR_COLOR_PRODUCTS,
            output_filename="top 10 products by sales bar graph.png",
            config=config
        )
        
        create_line_chart(
            data=top_products,
            title="TOP 10 PRODUCTS BY SALE",
            xlabel="product description",
            ylabel="Total Sales",
            color=config.LINE_COLOR,
            line_style=config.LINE_STYLE,
            marker=config.MARKER,
            marker_size=config.MARKER_SIZE,
            output_filename="top 10 products by sales line chart.png",
            config=config
        )
        
        create_bar_chart(
            data=top_customers,
            title="TOP 10 CUSTOMERS BY PURCHASE VOLUME",
            xlabel="CUSTOMER ID",
            ylabel="TOTAL SALES",
            color=config.BAR_COLOR_CUSTOMERS,
            output_filename="top 10 customeres by purchase volume bar graph.png",
            config=config
        )
        
        LOGGER.info("=" * 50)
        LOGGER.info("销售数据分析完成")
        LOGGER.info("=" * 50)
        
    except DataLoadError as e:
        LOGGER.error(f"数据加载失败: {e}")
        raise
    except DataCleanError as e:
        LOGGER.error(f"数据清洗失败: {e}")
        raise
    except AnalysisError as e:
        LOGGER.error(f"数据分析失败: {e}")
        raise
    except ChartError as e:
        LOGGER.error(f"图表生成失败: {e}")
        raise
    except Exception as e:
        LOGGER.error(f"未知错误: {e}")
        raise


if __name__ == "__main__":
    main()
