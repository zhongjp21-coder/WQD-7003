"""
Main runner for precipitation analysis pipeline
"""

import json
import argparse
from datetime import datetime

from functions.monthly_boundary_SPI import run_monthly_boundary_spi
from functions.growth_season_analysis import run_growth_analysis_pipeline
from functions.syrs_yield_analysis import run_complete_syrs_analysis
from functions.spi_syrs_correlation import run_complete_spi_syrs_analysis
from functions.cmip_climate_analysis import run_cmip_climate_analysis


def load_config(config_path: str, country: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f).get(country)


def main(country, run_tag):
    """Main entry point - runs monthly boundary precipitation pipeline"""
    parser = argparse.ArgumentParser(
        description='Run monthly precipitation analysis with administrative boundaries'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, country)
    config["run_tag"] = run_tag
    print("run_tag:",run_tag)

    # run_monthly_boundary_spi
    results = run_monthly_boundary_spi(config)
    #
    # Run the growth_season pipeline
    growth_results = run_growth_analysis_pipeline(config)
    #
    syrs_results = run_complete_syrs_analysis(config)
    correlation_results = run_complete_spi_syrs_analysis(config)  # default SPI_3

    # # CMIP analysis
    cmip_results = run_cmip_climate_analysis(config)

    # return results, growth_results



if __name__ == "__main__":
    # mainv("china", run_tag="01021450")
    run_tag = f"run_{datetime.now().strftime('%Y%m%d%H%M')}"
    main("malaysia", run_tag)

