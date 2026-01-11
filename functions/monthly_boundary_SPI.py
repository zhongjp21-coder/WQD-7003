"""
Complete pipeline for monthly precipitation calculation with administrative boundaries and SPI calculation
"""

import json
import pandas as pd

from datetime import datetime
from pathlib import Path

from config import get_country_paths
from utils.data_io import read_precipitation_data, save_processed_data
from utils.spatial_utils import filter_geographic_region
from utils.time_utils import process_time_series
from functions.precipitation_prepare import robust_missing_value_handling
from functions.precipitation_prepare import calculate_monthly_precipitation, clean_data
from functions.precipitation_prepare import add_administrative_regions, add_geographic_regions
from functions.precipitation_prepare import calculate_spi
import os
import requests
from tqdm import tqdm

# =================config=================
BASE_URL = "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/"
DATA_RANGE = [2017, 2022]
SAVE_DIR = "./datasets/Malaysia/rawdata/history_precipitation"


# =========================================

def download_cpc_fixed_range():

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"directory has been created: {SAVE_DIR}")

    start_year, end_year = DATA_RANGE
    print(f"In Progressing | Data source: {BASE_URL}")
    print(f"Time range: {start_year} to {end_year}")
    print(f"Saving location: {SAVE_DIR}")
    print("-" * 60)

    # 2. download data for each year
    for year in range(start_year, end_year + 1):
        file_name = f"precip.{year}.nc"
        file_url = f"{BASE_URL}{file_name}"
        save_path = os.path.join(SAVE_DIR, file_name)

        # check the file whether is existed
        if os.path.exists(save_path):
            print(f"[skip] file has existed: {file_name}")
            continue

        print(f"downloading: {file_name} ...")

        response = requests.get(file_url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # 写入文件并显示进度条
        with open(save_path, 'wb') as file, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)



    print("-" * 60)
    print("downloading completed。")



def run_monthly_boundary_spi(config: dict):
    """
    Run complete pipeline for monthly precipitation with administrative boundaries and SPI calculation

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    dict: Results containing monthly data, SPI data and paths
    """
    # Get country
    country = config.get('country', 'malaysia')

    # Generate run tag if not provided
    run_tag = config.get('run_tag')
    if run_tag is None:
        run_tag = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get file paths
    paths = get_country_paths(country, run_tag)

    print("=" * 60)
    print(f"MONTHLY BOUNDARY SPI PIPELINE")
    print("=" * 60)
    print(f"Country: {country.upper()}")
    print(f"Run tag: {run_tag}")
    print(f"Output directory: {paths['processed']}")
    print("=" * 60)

    # Get all parameters
    params = config.get('parameters', {})
    processing_options = config.get('processing_options', {})
    io_config = config.get('io_config', {})
    print(f"\n0. DOWNLOAD PRECIPITATION DATA FROM NOAA")
    download_cpc_fixed_range()


    print(f"\n1. READING DATA")
    print("-" * 40)

    # Determine data path
    data_path = io_config.get('data_path')
    if data_path is None:
        data_path = str(paths['rawdata'] / "history_precipitation")

    print(f"Data path: {data_path}")
    ds, missing_val, valid_range = read_precipitation_data(data_path)

    print(f"\n2. REGIONAL FILTERING")
    print("-" * 40)
    lat_range = tuple(params.get('lat_range', [0.5, 7.5]))
    lon_range = tuple(params.get('lon_range', [99.5, 120.5]))
    ds_sub = filter_geographic_region(ds, lat_range, lon_range)

    print(f"\n3. QUALITY CONTROL")
    print("-" * 40)
    qc_options = {
        'missing_val': missing_val,
        'valid_range': valid_range,
        'max_missing_ratio': params.get('max_missing_ratio', 0.3),
        'handle_negatives': processing_options.get('handle_negatives', 'zero'),
        'handle_high_values': processing_options.get('handle_high_values', 'nan'),
        'apply_interpolation': processing_options.get('apply_interpolation', True)
    }

    ds_clean, qc_report = robust_missing_value_handling(ds_sub, **qc_options)

    # Save quality control report
    report_path = paths['processed'] / "quality_control_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(qc_report, f, indent=2, default=str)
    print(f"✓ Quality control report saved to: {report_path}")

    print(f"\n4. TIME SERIES PROCESSING")
    print("-" * 40)
    dates = process_time_series(ds_clean)

    print(f"\n5. MONTHLY PRECIPITATION CALCULATION")
    print("-" * 40)
    df_monthly, grid_meta = calculate_monthly_precipitation(ds_clean, dates)

    print(f"\n6. REGION LABELING")
    print("-" * 40)
    print(paths['states_geojson'])
    if Path(paths['states_geojson']).exists():
        df_monthly = add_administrative_regions(
            df_monthly,
            str(paths['states_geojson']),
            grid_size=params.get('grid_size', 0.5)
        )
    else:
        print(f"Warning: Administrative boundaries file not found")
        df_monthly = add_geographic_regions(df_monthly)

    print(f"\n7. DATA CLEANING")
    print("-" * 40)
    df_clean = clean_data(df_monthly, min_valid_months=params.get('min_valid_months', 90))

    # Save cleaned monthly data
    output_format = io_config.get('output_format', 'csv')
    monthly_path = paths['processed']/f"monthly_precipitation_cleaned.{output_format}"
    save_processed_data(df_clean, monthly_path, format=output_format)

    print(f"\n8. SPI CALCULATION")
    print("-" * 40)
    spi_params = {
        'timescales': params.get('timescales', [1, 3, 6, 12]),
        'distribution': params.get('distribution', 'gamma')
    }

    df_spi = calculate_spi(df_clean, **spi_params)

    # Save SPI results
    spi_path = paths['processed']/ f"spi_results.{output_format}"
    save_processed_data(df_spi, spi_path, format=output_format)

    # Save metadata
    metadata = {
        'country': country,
        'run_tag': run_tag,
        'pipeline': 'monthly_boundary_SPI',
        'parameters': params,
        'processing_options': processing_options,
        'grid_metadata': grid_meta,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = paths['processed']/ "pipeline_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("✓ MONTHLY BOUNDARY SPI PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Results saved to:")
    print(f"  Monthly data (cleaned): {monthly_path}")
    print(f"  SPI results: {spi_path}")
    print(f"  QC report: {report_path}")
    print(f"  Metadata: {metadata_path}")
    print("=" * 60)

    # 统计各州指标
    print("df_spi",df_spi)
    print(df_spi.columns)
    state_summary = df_spi.groupby('region').agg(
        valid_grid=('grid_id', 'nunique'),
        total_value=('monthly_precip', 'count')
    ).reset_index()


    # calc average precipitation
    avg_precip = df_spi.groupby('region')['monthly_precip'].mean().reset_index()
    state_summary = pd.merge(state_summary, avg_precip.rename(columns={'monthly_precip': 'average precipitation(mm)'}),
                             on='region')

    # calc SPI succeed percentage
    def spi_success_rate(group):
        spi_cols = ['SPI_1', 'SPI_3', 'SPI_6', 'SPI_12']
        valid_spi = group[spi_cols].notna().any(axis=1).sum()
        return valid_spi / len(group) * 100 if len(group) > 0 else 0

    spi_rates = df_spi.groupby('region').apply(spi_success_rate).reset_index(name='SPI caculate succeed %')
    state_summary = pd.merge(state_summary, spi_rates, on='region')

    # 格式化输出
    state_summary = state_summary.round({
        'average precipitation(mm)': 2,
        'SPI caculate succeed %': 1
    })

    # ================= 新增：深度统计分析 =================
    # 1. 计算统计指标
    df_regional_stats = analyze_regional_drought_characteristics(df_spi)

    # 2. 保存统计结果到 CSV
    stats_path = paths['processed'] / "regional_drought_statistics.csv"
    df_regional_stats.to_csv(stats_path, index=False)
    print(f"✓ 区域干旱统计表已保存: {stats_path}")

    img_save_dir = str(paths['processed'])  # 使用你的输出目录
    # 1. 画 "极端干旱" (Extreme) 频率图
    plot_drought_severity_heatmap(df_regional_stats, severity='Extreme', save_dir=img_save_dir)

    # 2. 画 "严重干旱" (Severe) 频率图
    plot_drought_severity_heatmap(df_regional_stats, severity='Severe', save_dir=img_save_dir)

    # 4. 画趋势分析图 (带星号)
    plot_trend_heatmap(df_regional_stats, save_dir=img_save_dir)

    # D. [指定] 时空演变热力图 (只画 SPI-3)
    plot_temporal_heatmap(df_spi, timescale="SPI_3", save_dir=img_save_dir)


    # ====================================================

    return {
        'monthly_data': df_clean,
        'spi_data': df_spi,
        'grid_metadata': grid_meta,
        'qc_report': qc_report,
        'metadata': metadata,
        'paths': paths
    }


import pandas as pd
import numpy as np
from scipy import stats


def analyze_regional_drought_characteristics(df_spi):
    """
    对各州的 SPI 数据进行深度统计分析：干旱频率、强度和趋势
    """
    print("=" * 60)
    print("正在进行区域干旱特征统计分析...")
    print("=" * 60)

    stats_list = []

    # 获取所有 SPI 列 (例如 SPI_1, SPI_3...)
    spi_cols = [c for c in df_spi.columns if 'SPI_' in c]

    # 按州分组遍历
    for region, group in df_spi.groupby('region'):
        region_stats = {'Region': region}
        print("analyzing...",group)
        for col in spi_cols:
            # 提取有效数据
            valid_data = group[col].dropna()
            if len(valid_data) == 0:
                continue

            # --- 1. 干旱频率分析 (Frequency) - 基于图片标准 ---
            total_months = len(valid_data)

            # 1. Mild drought: 0 to -0.99
            # 逻辑：小于等于0 且 大于 -1.0
            cnt_mild = ((valid_data <= 0) & (valid_data > -1.0)).sum()

            # 2. Moderate drought: -1.00 to -1.49
            # 逻辑：小于等于 -1.0 且 大于 -1.5
            cnt_mod = ((valid_data <= -1.0) & (valid_data > -1.5)).sum()

            # 3. Severe drought: -1.50 to -1.99 (修正了图片的笔误 1.50)
            # 逻辑：小于等于 -1.5 且 大于 -2.0
            cnt_sev = ((valid_data <= -1.5) & (valid_data > -2.0)).sum()

            # 4. Extreme drought: <= -2.00
            # 逻辑：小于等于 -2.0
            cnt_ext = (valid_data <= -2.0).sum()

            # 计算百分比并存入字典 (保留2位小数)
            region_stats[f'{col}_Mild(%)'] = round((cnt_mild / total_months) * 100, 2)
            region_stats[f'{col}_Moderate(%)'] = round((cnt_mod / total_months) * 100, 2)
            region_stats[f'{col}_Severe(%)'] = round((cnt_sev / total_months) * 100, 2)
            region_stats[f'{col}_Extreme(%)'] = round((cnt_ext / total_months) * 100, 2)

            # --- 2. 极值分析 (Extremes) ---
            # 历史上最干旱的 SPI 值
            region_stats[f'{col}_Min'] = round(valid_data.min(), 2)

            # --- 3. 趋势分析 (Trend) - 针对 2017-2022 短期数据优化 ---
            # 只有 6 年数据，我们计算 "SPI 每年的变化斜率"
            # 门槛设为 > 24 个月 (至少有2年有效数据才计算趋势)
            if len(valid_data) > 24:
                # 1. 提取时间
                dates = pd.to_datetime(group.loc[valid_data.index, 'time'])

                # 2. 将时间转换为 "相对年份" (Relative Years)
                # 例如: 2017-01-01 是 0.0 年, 2018-01-01 是 1.0 年
                start_date = dates.min()
                x_years = (dates - start_date).dt.days / 365.25

                # 3. 线性回归计算
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_years, valid_data)

                # 4. 结果存储
                # 直接存储 slope，代表 "SPI Value Change Per Year" (每年变化量)
                region_stats[f'{col}_Trend(per_year)'] = round(slope, 4)

                # 显著性判断 (由于样本量少，P值可能会偏大，但这如实反映了短期趋势的不确定性)
                region_stats[f'{col}_Trend_Signif'] = p_value < 0.05

            else:
                region_stats[f'{col}_Trend(per_year)'] = np.nan
                region_stats[f'{col}_Trend_Signif'] = False

        stats_list.append(region_stats)

    # 转换为 DataFrame
    df_stats = pd.DataFrame(stats_list)

    return df_stats


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_drought_severity_heatmap(df_stats, severity='Extreme', save_dir=None):
    """
    绘制特定干旱等级频率的热力图

    Parameters:
    -----------
    df_stats : pd.DataFrame
        由 analyze_regional_drought_characteristics 生成的统计表
    severity : str
        要可视化的干旱等级，可选: 'Mild', 'Moderate', 'Severe', 'Extreme'
    """
    # 构造列名后缀，例如 "_Extreme(%)"
    metric_suffix = f'_{severity}(%)'

    # 筛选相关列
    target_cols = [c for c in df_stats.columns if metric_suffix in c]

    if not target_cols:
        print(f"⚠️ 未找到包含 {metric_suffix} 的列，请检查数据或 severity 参数。")
        print(f"可用列名示例: {df_stats.columns[:5]}")
        return

    # 准备绘图数据：设置 Region 为索引
    plot_data = df_stats.set_index('Region')[target_cols]

    # 简化列名 (去掉后缀，只保留 SPI_1, SPI_3 等)
    plot_data.columns = [c.replace(metric_suffix, '') for c in plot_data.columns]

    # --- 排序优化 ---
    # 按所有尺度的平均频率排序，让干旱严重的地区排在上面
    plot_data['mean'] = plot_data.mean(axis=1)
    plot_data = plot_data.sort_values('mean', ascending=False).drop(columns='mean')

    # --- 绘图 ---
    # 根据严重程度选择颜色: 轻度用黄色调，极端用红色/紫色调
    if severity in ['Severe', 'Extreme']:
        cmap = 'Reds'
    else:
        cmap = 'YlOrBr'

    plt.figure(figsize=(10, len(plot_data) * 0.5 + 2))

    ax = sns.heatmap(plot_data,
                     annot=True,
                     cmap=cmap,
                     fmt='.1f',
                     linewidths=.5,
                     cbar_kws={'label': f'Frequency of {severity} Drought (%)'})

    plt.title(f'Regional Drought Analysis: {severity} Drought Frequency', fontsize=14)
    plt.ylabel('Region (State)', fontsize=12)
    plt.xlabel('SPI Timescale', fontsize=12)
    plt.tight_layout()

    if save_dir:
        import os
        filename = f"heatmap_{severity}_frequency.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 热力图已保存: {save_path}")



def plot_trend_heatmap(df_stats, save_dir=None):
    """
    绘制 SPI 变化趋势热力图，并标注显著性
    """
    # 筛选趋势列
    trend_cols = [c for c in df_stats.columns if 'Trend(per_year)' in c]

    if not trend_cols:
        print("⚠️ 未找到趋势数据列")
        return

    # 准备主数据 (斜率)
    slope_data = df_stats.set_index('Region')[trend_cols]
    slope_data.columns = [c.replace('_Trend(per_year)', '') for c in slope_data.columns]  # 简化列名

    # 准备显著性数据 (用于打星号)
    # 找到对应的 Signif 列
    annot_data = slope_data.copy()
    for col in slope_data.columns:  # col 是 "SPI_1"
        signif_col_name = f"{col}_Trend_Signif"
        # 如果显著(True)，标记为 "*"，否则为空
        is_signif = df_stats.set_index('Region')[signif_col_name]
        annot_data[col] = is_signif.apply(
            lambda x: f"{x:.4f} *" if x is True else f"{x:.4f}")  # 这里只用于显示，但heatmap annot如果要自定义格式比较麻烦

    # --- 绘图 ---
    plt.figure(figsize=(12, len(slope_data) * 0.5 + 2))

    # 使用红蓝配色：红色代表变干(负值)，蓝色代表变湿(正值)
    # center=0 确保 0 值是白色的
    ax = sns.heatmap(slope_data,
                     annot=True,  # 这里直接显示数值
                     cmap='RdBu',
                     center=0,
                     fmt='.3f',
                     linewidths=.5,
                     cbar_kws={'label': 'SPI Change per Year (Slope)'})

    # --- 手动添加星号标记 ---
    # 遍历每个单元格，检查是否显著
    for y in range(slope_data.shape[0]):
        for x in range(slope_data.shape[1]):
            region = slope_data.index[y]
            col_name = slope_data.columns[x]  # SPI_1, SPI_3...

            # 找到对应的 Signif 值
            original_col = f"{col_name}_Trend_Signif"
            is_significant = df_stats[df_stats['Region'] == region][original_col].values[0]

            if is_significant:
                # 在格子中心稍微偏上的位置画一个星号
                plt.text(x + 0.5, y + 0.3, '★',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='black', fontsize=14, weight='bold')

    plt.title('Short-term SPI Trends (2017-2022)\n(★ indicates statistical significance p<0.05)', fontsize=14)
    plt.ylabel('Region', fontsize=12)
    plt.tight_layout()

    if save_dir:
        import os
        save_path = os.path.join(save_dir, "heatmap_spi_trends.png")
        plt.savefig(save_path, dpi=300)
        print(f"✓ 趋势热力图已保存: {save_path}")





def plot_temporal_heatmap(df_spi, timescale='SPI_3', save_dir=None):
    """
    绘制 SPI 时空演变热力图 (优化版)
    优化点：
    1. 时间标签格式化 (YYYY-mm)
    2. 按平均 SPI 值对州进行排序 (最干旱的排上面)
    3. 添加年份分割线
    """

    # --- 1. 数据聚合 ---
    # 确保时间列是 datetime 格式
    df_spi['time'] = pd.to_datetime(df_spi['time'])

    # 聚合：算出每个州每个月的平均 SPI
    region_time_avg = df_spi.groupby(['region', 'time'])[timescale].mean().reset_index()

    # --- 2. 数据透视 ---
    pivot_df = region_time_avg.pivot(index='region', columns='time', values=timescale)

    # --- 优化 A: 智能排序 (Sort by Dryness) ---
    # 计算每个州的平均 SPI 值，从小到大排序 (数值越小越干)
    # 这样最红(最干)的州会跑去最上面，视觉上会有"重灾区"的感觉
    mean_val = pivot_df.mean(axis=1).sort_values()
    pivot_df = pivot_df.reindex(mean_val.index)

    # --- 3. 绘图 ---
    plt.figure(figsize=(16, 8))  # 稍微拉宽一点，适应时间轴

    # 绘制热力图
    ax = sns.heatmap(pivot_df, cmap='RdBu', center=0, vmin=-2.5, vmax=2.5,
                     cbar_kws={'label': f'{timescale} Value', 'shrink': 0.8})

    # --- 优化 B: X 轴时间标签美化 ---
    # 获取原本的 x 轴刻度位置 (0, 1, 2...)
    # 提取对应的时间标签
    x_dates = pivot_df.columns

    new_ticks = []
    new_labels = []

    for i, date in enumerate(x_dates):
        # 策略：每年只显示 1月 和 7月 的标签
        if date.month in [1, 7]:
            new_ticks.append(i + 0.5)  # +0.5 让标签居中显示在格子上
            new_labels.append(date.strftime('%Y-%m'))  # 格式化为 2017-01

    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=10)

    # --- 优化 C: 添加年份分割线 ---
    # 在每年的 1月1日 位置画一条竖虚线
    for i, date in enumerate(x_dates):
        if date.month == 1 and i > 0:  # i>0 避免在最左边画线
            plt.axvline(i, color='black', linestyle='--', linewidth=0.7, alpha=0.5)

    # 标题和标签
    plt.title(f'Spatiotemporal Evolution of {timescale} (2017-2022)\n(Sorted by Mean Dryness: Driest Regions on Top)',
              fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Region', fontsize=12)

    plt.tight_layout()

    # 保存逻辑
    if save_dir:
        # 文件名动态化，防止覆盖其他尺度的图
        filename = f"heatmap_temporal_{timescale}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 优化版热力图已保存: {save_path}")


