"""
销售数据分析模块
对在线销售数据集进行数据清洗、分析和可视化。

功能特性:
- 数据加载与清洗
- 热门商品分析
- 优质客户分析
- 图表生成与保存
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SalesAnalysisError(Exception):
    """销售数据分析模块自定义异常基类"""
    pass


class DataFileNotFoundError(SalesAnalysisError):
    """数据文件未找到异常"""
    pass


class DataFormatError(SalesAnalysisError):
    """数据格式错误异常"""
    pass


class EmptyDataError(SalesAnalysisError):
    """空数据异常"""
    pass


@dataclass
class Config:
    """全局配置类"""

    DATA_FILE_PATH: Path = Path("./DATASET/online_sales_dataset.csv")
    OUTPUT_DIR: Path = Path("./visuals/")
    CLEAN_COLUMNS: Sequence[str] = ("CustomerID", "ShippingCost", "WarehouseLocation")
    FIGURE_SIZE: tuple[int, int] = (10, 6)
    BAR_COLOR_PRODUCTS: str = "teal"
    BAR_COLOR_CUSTOMERS: str = "coral"
    LINE_COLOR: str = "blue"
    LINE_STYLE: str = "--"
    MARKER_STYLE: str = "o"
    MARKER_SIZE: int = 10
    TOP_N: int = 10

    def __post_init__(self) -> None:
        """初始化后确保输出目录存在"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path: Path) -> DataFrame:
    """
    从CSV文件加载销售数据。

    参数:
        file_path: 数据文件路径

    返回:
        加载后的DataFrame

    异常:
        DataFileNotFoundError: 文件不存在
        DataFormatError: 数据格式错误
    """
    try:
        if not file_path.exists():
            raise DataFileNotFoundError(f"数据文件不存在: {file_path.absolute()}")

        logger.info(f"正在加载数据文件: {file_path}")
        df = pd.read_csv(file_path)

        if df.empty:
            raise EmptyDataError("加载的数据为空")

        logger.debug(f"数据列名: {df.columns.tolist()}")
        logger.info(f"成功加载 {len(df)} 条记录")
        return df

    except pd.errors.EmptyDataError as e:
        raise DataFormatError(f"CSV文件格式错误或为空: {e}") from e
    except pd.errors.ParserError as e:
        raise DataFormatError(f"CSV文件解析错误: {e}") from e
    except Exception as e:
        raise SalesAnalysisError(f"加载数据时发生未知错误: {e}") from e


def clean_data(df: DataFrame, clean_columns: Sequence[str]) -> DataFrame:
    """
    清洗数据：删除指定列含空值的行，转换日期格式。

    参数:
        df: 原始DataFrame
        clean_columns: 需要清洗空值的列名列表

    返回:
        清洗后的DataFrame

    异常:
        EmptyDataError: 清洗后数据为空
    """
    try:
        logger.info("开始数据清洗")
        logger.debug(f"清洗前列空值统计:\n{df[list(clean_columns)].isna().sum()}")

        df_cleaned = df.dropna(subset=list(clean_columns)).copy()

        missing_after = df_cleaned.isna().sum()
        logger.debug(f"清洗后列空值统计:\n{missing_after}")

        if "InvoiceDate" in df_cleaned.columns:
            df_cleaned["InvoiceDate"] = pd.to_datetime(
                df_cleaned["InvoiceDate"],
                errors="coerce"
            )

        if "CustomerID" in df_cleaned.columns:
            df_cleaned["CustomerID"] = df_cleaned["CustomerID"].astype("Int64")

        df_cleaned["Total Sales"] = (
            df_cleaned["Quantity"] *
            df_cleaned["UnitPrice"] *
            (1 - df_cleaned["Discount"])
        )

        if df_cleaned.empty:
            raise EmptyDataError("数据清洗后结果为空")

        logger.info(f"数据清洗完成，剩余 {len(df_cleaned)} 条记录")
        return df_cleaned

    except KeyError as e:
        raise DataFormatError(f"数据缺少必要的列: {e}") from e
    except Exception as e:
        raise SalesAnalysisError(f"数据清洗时发生未知错误: {e}") from e


def analyze_top_products(df: DataFrame, top_n: int = 10) -> Series:
    """
    分析销量最高的前N个商品。

    参数:
        df: 清洗后的DataFrame
        top_n: 返回前N个商品

    返回:
        按总销售额排序的商品Series
    """
    try:
        logger.info(f"分析销量前 {top_n} 商品")

        if "Description" not in df.columns or "Total Sales" not in df.columns:
            raise DataFormatError("数据缺少商品分析所需列")

        top_products = (
            df.groupby("Description")["Total Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )

        logger.info("销量前10商品分析完成")
        print("TOP 10 PRODUCTS")
        print(top_products)
        return top_products

    except Exception as e:
        raise SalesAnalysisError(f"商品分析时发生错误: {e}") from e


def analyze_top_customers(df: DataFrame, top_n: int = 10) -> Series:
    """
    分析购买力最高的前N个客户。

    参数:
        df: 清洗后的DataFrame
        top_n: 返回前N个客户

    返回:
        按总购买额排序的客户Series
    """
    try:
        logger.info(f"分析购买力前 {top_n} 客户")

        if "CustomerID" not in df.columns or "Total Sales" not in df.columns:
            raise DataFormatError("数据缺少客户分析所需列")

        top_customers = (
            df.groupby("CustomerID")["Total Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )

        logger.info("客户分析完成")
        print("TOP 10 CUSTOMERS ")
        print(top_customers)
        return top_customers

    except Exception as e:
        raise SalesAnalysisError(f"客户分析时发生错误: {e}") from e


def save_chart(fig: plt.Figure, output_path: Path) -> None:
    """
    保存图表并关闭以防止内存泄漏。

    参数:
        fig: matplotlib Figure对象
        output_path: 输出文件路径
    """
    try:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"图表已保存: {output_path}")
    except Exception as e:
        logger.warning(f"保存图表失败 {output_path}: {e}")


def create_bar_chart(
    data: Series,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    config: Config,
    filename: str
) -> None:
    """
    创建并保存柱状图。

    参数:
        data: 要可视化的数据Series
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        color: 柱状图颜色
        config: 配置对象
        filename: 输出文件名
    """
    try:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

        data.plot(kind="bar", color=color, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = config.OUTPUT_DIR / filename
        save_chart(fig, output_path)
        logger.info(f"柱状图已生成: {filename}")

    except Exception as e:
        logger.error(f"创建柱状图失败: {e}")
        raise SalesAnalysisError(f"创建柱状图失败: {e}") from e


def create_line_chart(
    data: Series,
    title: str,
    xlabel: str,
    ylabel: str,
    config: Config,
    filename: str
) -> None:
    """
    创建并保存折线图。

    参数:
        data: 要可视化的数据Series
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        config: 配置对象
        filename: 输出文件名
    """
    try:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        product_names = data.index.tolist()

        data.plot(
            kind="line",
            color=config.LINE_COLOR,
            linestyle=config.LINE_STYLE,
            marker=config.MARKER_STYLE,
            markersize=config.MARKER_SIZE,
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(product_names)))
        ax.set_xticklabels(product_names, rotation=90)
        ax.grid(True)
        plt.tight_layout()

        output_path = config.OUTPUT_DIR / filename
        save_chart(fig, output_path)
        logger.info(f"折线图已生成: {filename}")

    except Exception as e:
        logger.error(f"创建折线图失败: {e}")
        raise SalesAnalysisError(f"创建折线图失败: {e}") from e


def main() -> None:
    """主函数：执行完整的销售数据分析流程"""
    config = Config()

    try:
        logger.info("=== 开始销售数据分析 ===")

        df = load_data(config.DATA_FILE_PATH)

        print("列名:", df.columns.tolist())
        print("数据信息:")
        print(df.info())
        print("统计信息\n", df.describe())
        print("空值统计:\n", df.isna().sum())

        df_cleaned = clean_data(df, config.CLEAN_COLUMNS)

        print("清洗后空值统计:\n", df_cleaned.isna().sum())

        top_products = analyze_top_products(df_cleaned, config.TOP_N)

        create_bar_chart(
            data=top_products,
            title="TOP 10 PRODUCTS BY SALE",
            xlabel="product description",
            ylabel="Total Sales",
            color=config.BAR_COLOR_PRODUCTS,
            config=config,
            filename="top_products_bar.png"
        )

        create_line_chart(
            data=top_products,
            title="TOP 10 PRODUCTS BY SALE",
            xlabel="product description",
            ylabel="Total Sales",
            config=config,
            filename="top_products_line.png"
        )

        top_customers = analyze_top_customers(df_cleaned, config.TOP_N)

        create_bar_chart(
            data=top_customers,
            title="TOP 10 CUSTOMERS BY PURCHASE VOLUME",
            xlabel="CUSTOMER ID",
            ylabel="TOTAL SALES",
            color=config.BAR_COLOR_CUSTOMERS,
            config=config,
            filename="top_customers_bar.png"
        )

        logger.info("=== 销售数据分析完成 ===")
        logger.info(f"所有图表已保存至: {config.OUTPUT_DIR.absolute()}")

    except SalesAnalysisError as e:
        logger.error(f"业务错误: {e}", exc_info=True)
        exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
