"""Generate a per-symbol OHLCV quality report for the configured universe."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
from src.data_validator import DataValidator

ROOT = Path(__file__).resolve().parents[1]


def load_universe(path: Path = ROOT / "data" / "universe.yaml") -> list[str]:
    """Read symbol identifiers from the project universe YAML."""
    with path.open("r", encoding="utf-8") as stream:
        document: dict[str, Any] = yaml.safe_load(stream) or {}
    return [str(stock["symbol"]) for stock in document.get("stocks", []) if stock.get("symbol")]


def main() -> int:
    """Validate configured symbols, save CSV report and print summary metrics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    validator = DataValidator(ROOT / "config.yaml")
    report = validator.generate_quality_report(load_universe())
    output = ROOT / "reports" / "data_quality_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f"Symbols checked: {len(report)}")
    print(f"Rows available: {int(report['total_rows'].sum()) if not report.empty else 0}")
    print(f"Mean quality score: {report['quality_score'].mean():.2f}" if not report.empty else "Mean quality score: n/a")
    print(f"Symbols with no data: {int((report['total_rows'] == 0).sum()) if not report.empty else 0}")
    print(f"Report saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
