import sys
from collections import defaultdict
from datetime import datetime
from functools import reduce
from importlib import import_module, metadata, reload
from pathlib import Path
from typing import Annotated, Any, Optional

from typer import Argument, Option, Typer

from prosperity3bt.data import has_day_data
from prosperity3bt.file_reader import FileReader, FileSystemReader, PackageResourcesReader
from prosperity3bt.models import BacktestResult, TradeMatchingMode
from prosperity3bt.open import open_visualizer
from prosperity3bt.runner import run_backtest

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.SQUID_INK: {
        "take_width": 0.0,
        "clear_width": 0.7989037527218122,
        "prevent_adverse": True,
        "adverse_volume": 12,
        "reversion_beta": -0.262018206155453,
        "disregard_edge": 0.0,
        "join_edge": 8.205011595727127,
        "default_edge": 1.0,
        "z_rolling_window": 5000,     # number of ticks to consider for the rolling average
        "zscore_threshold": 3.1571449177636266,  # threshold for taking advantage of large swings
    },
    Product.KELP: {  # added KELP with similar settings as SQUID_INK
        "take_width": 3.0,
        "clear_width": 0.0,
        "prevent_adverse": True,
        "adverse_volume": 7,
        "reversion_beta": -0.231709901360756,
        "disregard_edge": 4.0,
        "join_edge": -1.8535170129297578,
        "default_edge": 1.0,
    },
    Product.CROISSANTS: {
        "take_width": 1.0,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
    },
    Product.JAMS: {
        "take_width": 1.0,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
    },
    Product.DJEMBES: {
        "take_width": 1.0,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
    },
    Product.PICNIC_BASKET1: {
        "spread_threshold": 10.0,
        "max_trade_size": 11,
    },
    Product.PICNIC_BASKET2: {
        "spread_threshold": 2,
        "max_trade_size": 5,
    }
}

BASKET_COMPOSITIONS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1,
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
    }
}


def parse_algorithm(algorithm: Path) -> Any:
    sys.path.append(str(algorithm.parent))
    return import_module(algorithm.stem)


def parse_data(data_root: Optional[Path]) -> FileReader:
    if data_root is not None:
        return FileSystemReader(data_root)
    else:
        return PackageResourcesReader()


def parse_days(file_reader: FileReader, days: list[str]) -> list[tuple[int, int]]:
    parsed_days = []

    for arg in days:
        if "-" in arg:
            round_num, day_num = map(int, arg.split("-", 1))

            if not has_day_data(file_reader, round_num, day_num):
                print(f"Warning: no data found for round {round_num} day {day_num}")
                continue

            parsed_days.append((round_num, day_num))
        else:
            round_num = int(arg)

            parsed_days_in_round = []
            for day_num in range(-5, 6):
                if has_day_data(file_reader, round_num, day_num):
                    parsed_days_in_round.append((round_num, day_num))

            if len(parsed_days_in_round) == 0:
                print(f"Warning: no data found for round {round_num}")
                continue

            parsed_days.extend(parsed_days_in_round)

    if len(parsed_days) == 0:
        print("Error: did not find data for any requested round/day")
        sys.exit(1)

    return parsed_days


def parse_out(out: Optional[Path], no_out: bool) -> Optional[Path]:
    if out is not None:
        return out

    if no_out:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path.cwd() / "backtests" / f"{timestamp}.log"


def print_day_summary(result: BacktestResult) -> None:
    last_timestamp = result.activity_logs[-1].timestamp

    product_lines = []
    total_profit = 0

    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break

        product = row.columns[2]
        profit = row.columns[-1]

        product_lines.append(f"{product}: {profit:,.0f}")
        total_profit += profit

    print(*reversed(product_lines), sep="\n")
    print(f"Total profit: {total_profit:,.0f}")


def merge_results(
    a: BacktestResult, b: BacktestResult, merge_profit_loss: bool, merge_timestamps: bool
) -> BacktestResult:
    sandbox_logs = a.sandbox_logs[:]
    activity_logs = a.activity_logs[:]
    trades = a.trades[:]

    if merge_timestamps:
        a_last_timestamp = a.activity_logs[-1].timestamp
        timestamp_offset = a_last_timestamp + 100
    else:
        timestamp_offset = 0

    sandbox_logs.extend([row.with_offset(timestamp_offset) for row in b.sandbox_logs])
    trades.extend([row.with_offset(timestamp_offset) for row in b.trades])

    if merge_profit_loss:
        profit_loss_offsets = defaultdict(float)
        for row in reversed(a.activity_logs):
            if row.timestamp != a_last_timestamp:
                break

            profit_loss_offsets[row.columns[2]] = row.columns[-1]

        activity_logs.extend(
            [row.with_offset(timestamp_offset, profit_loss_offsets[row.columns[2]]) for row in b.activity_logs]
        )
    else:
        activity_logs.extend([row.with_offset(timestamp_offset, 0) for row in b.activity_logs])

    return BacktestResult(a.round_num, a.day_num, sandbox_logs, activity_logs, trades)


def write_output(output_file: Path, merged_results: BacktestResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w+", encoding="utf-8") as file:
        file.write("Sandbox logs:\n")
        for row in merged_results.sandbox_logs:
            file.write(str(row))

        file.write("\n\n\nActivities log:\n")
        file.write(
            "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
        )
        file.write("\n".join(map(str, merged_results.activity_logs)))

        file.write("\n\n\n\n\nTrade History:\n")
        file.write("[\n")
        file.write(",\n".join(map(str, merged_results.trades)))
        file.write("]")


def print_overall_summary(results: list[BacktestResult]) -> None:
    print("Profit summary:")

    total_profit = 0
    for result in results:
        last_timestamp = result.activity_logs[-1].timestamp

        profit = 0
        for row in reversed(result.activity_logs):
            if row.timestamp != last_timestamp:
                break

            profit += row.columns[-1]

        print(f"Round {result.round_num} day {result.day_num}: {profit:,.0f}")
        total_profit += profit

    print(f"Total profit: {total_profit:,.0f}")
    with open("profit_log.txt", "a") as file:
        line = f"Tot Profit: {total_profit} with reversion_beta: {PARAMS[Product.KELP]["reversion_beta"]}, take_width: {PARAMS[Product.KELP]["take_width"]}, clear_width: {PARAMS[Product.KELP]["clear_width"]}, adverse_volume: {PARAMS[Product.KELP]["adverse_volume"]}\n"
        file.write(line)


def format_path(path: Path) -> str:
    cwd = Path.cwd()
    if path.is_relative_to(cwd):
        return str(path.relative_to(cwd))
    else:
        return str(path)


def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity3bt {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm to backtest.", show_default=False, exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    days: Annotated[list[str], Argument(help="The days to backtest on. <round>-<day> for a single day, <round> for all days in a round.", show_default=False)],
    
    # Basket-level
    PICNIC_BASKET1_spread_threshold: Annotated[Optional[float], Option("--PICNIC_BASKET1.spread_threshold")] = None,
    PICNIC_BASKET1_max_trade_size: Annotated[Optional[int], Option("--PICNIC_BASKET1.max_trade_size")] = None,

    PICNIC_BASKET2_spread_threshold: Annotated[Optional[float], Option("--PICNIC_BASKET2.spread_threshold")] = None,
    PICNIC_BASKET2_max_trade_size: Annotated[Optional[int], Option("--PICNIC_BASKET2.max_trade_size")] = None,

    # CROISSANTS
    CROISSANTS_take_width: Annotated[Optional[float], Option("--CROISSANTS.take_width")] = None,
    CROISSANTS_clear_width: Annotated[Optional[float], Option("--CROISSANTS.clear_width")] = None,
    CROISSANTS_adverse_volume: Annotated[Optional[int], Option("--CROISSANTS.adverse_volume")] = None,
    CROISSANTS_disregard_edge: Annotated[Optional[float], Option("--CROISSANTS.disregard_edge")] = None,
    CROISSANTS_join_edge: Annotated[Optional[float], Option("--CROISSANTS.join_edge")] = None,
    CROISSANTS_default_edge: Annotated[Optional[float], Option("--CROISSANTS.default_edge")] = None,

    # JAMS
    JAMS_take_width: Annotated[Optional[float], Option("--JAMS.take_width")] = None,
    JAMS_clear_width: Annotated[Optional[float], Option("--JAMS.clear_width")] = None,
    JAMS_adverse_volume: Annotated[Optional[int], Option("--JAMS.adverse_volume")] = None,
    JAMS_disregard_edge: Annotated[Optional[float], Option("--JAMS.disregard_edge")] = None,
    JAMS_join_edge: Annotated[Optional[float], Option("--JAMS.join_edge")] = None,
    JAMS_default_edge: Annotated[Optional[float], Option("--JAMS.default_edge")] = None,

    # DJEMBES
    DJEMBES_take_width: Annotated[Optional[float], Option("--DJEMBES.take_width")] = None,
    DJEMBES_clear_width: Annotated[Optional[float], Option("--DJEMBES.clear_width")] = None,
    DJEMBES_adverse_volume: Annotated[Optional[int], Option("--DJEMBES.adverse_volume")] = None,
    DJEMBES_disregard_edge: Annotated[Optional[float], Option("--DJEMBES.disregard_edge")] = None,
    DJEMBES_join_edge: Annotated[Optional[float], Option("--DJEMBES.join_edge")] = None,
    DJEMBES_default_edge: Annotated[Optional[float], Option("--DJEMBES.default_edge")] = None,


    merge_pnl: Annotated[bool, Option("--merge-pnl", help="Merge profit and loss across days.")] = False,
    vis: Annotated[bool, Option("--vis", help="Open backtest results in https://jmerle.github.io/imc-prosperity-3-visualizer/ when done.")] = False,
    out: Annotated[Optional[Path], Option(help="File to save output log to (defaults to backtests/<timestamp>.log).", show_default=False, dir_okay=False, resolve_path=True)] = None,
    no_out: Annotated[bool, Option("--no-out", help="Skip saving output log.")] = False,
    data: Annotated[Optional[Path], Option(help="Path to data directory. Must look similar in structure to https://github.com/jmerle/imc-prosperity-3-backtester/tree/master/prosperity3bt/resources.", show_default=False, exists=True, file_okay=False, dir_okay=True, resolve_path=True)] = None,
    print_output: Annotated[bool, Option("--print", help="Print the trader's output to stdout while it's running.")] = False,
    match_trades: Annotated[TradeMatchingMode, Option(help="How to match orders against market trades. 'all' matches trades with prices equal to or worse than your quotes, 'worse' matches trades with prices worse than your quotes, 'none' does not match trades against orders at all.")] = TradeMatchingMode.all,
    no_progress: Annotated[bool, Option("--no-progress", help="Don't show progress bars.")] = False,
    original_timestamps: Annotated[bool, Option("--original-timestamps", help="Preserve original timestamps in output log rather than making them increase across days.")] = False,
    version: Annotated[bool, Option("--version", "-v", help="Show the program's version number and exit.", is_eager=True, callback=version_callback)] = False,
) -> None:  # fmt: skip
    if out is not None and no_out:
        print("Error: --out and --no-out are mutually exclusive")
        sys.exit(1)

    # Basket 1
    if PICNIC_BASKET1_spread_threshold is not None:
        PARAMS[Product.PICNIC_BASKET1]["spread_threshold"] = PICNIC_BASKET1_spread_threshold
    if PICNIC_BASKET1_max_trade_size is not None:
        PARAMS[Product.PICNIC_BASKET1]["max_trade_size"] = PICNIC_BASKET1_max_trade_size

    # Basket 2
    if PICNIC_BASKET2_spread_threshold is not None:
        PARAMS[Product.PICNIC_BASKET2]["spread_threshold"] = PICNIC_BASKET2_spread_threshold
    if PICNIC_BASKET2_max_trade_size is not None:
        PARAMS[Product.PICNIC_BASKET2]["max_trade_size"] = PICNIC_BASKET2_max_trade_size

    # CROISSANTS
    if CROISSANTS_take_width is not None:
        PARAMS[Product.CROISSANTS]["take_width"] = CROISSANTS_take_width
    if CROISSANTS_clear_width is not None:
        PARAMS[Product.CROISSANTS]["clear_width"] = CROISSANTS_clear_width
    if CROISSANTS_adverse_volume is not None:
        PARAMS[Product.CROISSANTS]["adverse_volume"] = CROISSANTS_adverse_volume
    if CROISSANTS_disregard_edge is not None:
        PARAMS[Product.CROISSANTS]["disregard_edge"] = CROISSANTS_disregard_edge
    if CROISSANTS_join_edge is not None:
        PARAMS[Product.CROISSANTS]["join_edge"] = CROISSANTS_join_edge
    if CROISSANTS_default_edge is not None:
        PARAMS[Product.CROISSANTS]["default_edge"] = CROISSANTS_default_edge

    # JAMS
    if JAMS_take_width is not None:
        PARAMS[Product.JAMS]["take_width"] = JAMS_take_width
    if JAMS_clear_width is not None:
        PARAMS[Product.JAMS]["clear_width"] = JAMS_clear_width
    if JAMS_adverse_volume is not None:
        PARAMS[Product.JAMS]["adverse_volume"] = JAMS_adverse_volume
    if JAMS_disregard_edge is not None:
        PARAMS[Product.JAMS]["disregard_edge"] = JAMS_disregard_edge
    if JAMS_join_edge is not None:
        PARAMS[Product.JAMS]["join_edge"] = JAMS_join_edge
    if JAMS_default_edge is not None:
        PARAMS[Product.JAMS]["default_edge"] = JAMS_default_edge

    # DJEMBES
    if DJEMBES_take_width is not None:
        PARAMS[Product.DJEMBES]["take_width"] = DJEMBES_take_width
    if DJEMBES_clear_width is not None:
        PARAMS[Product.DJEMBES]["clear_width"] = DJEMBES_clear_width
    if DJEMBES_adverse_volume is not None:
        PARAMS[Product.DJEMBES]["adverse_volume"] = DJEMBES_adverse_volume
    if DJEMBES_disregard_edge is not None:
        PARAMS[Product.DJEMBES]["disregard_edge"] = DJEMBES_disregard_edge
    if DJEMBES_join_edge is not None:
        PARAMS[Product.DJEMBES]["join_edge"] = DJEMBES_join_edge
    if DJEMBES_default_edge is not None:
        PARAMS[Product.DJEMBES]["default_edge"] = DJEMBES_default_edge



    try:
        trader_module = parse_algorithm(algorithm)
    except ModuleNotFoundError as e:
        print(f"{algorithm} is not a valid algorithm file: {e}")
        sys.exit(1)

    if not hasattr(trader_module, "Trader"):
        print(f"{algorithm} does not expose a Trader class")
        sys.exit(1)

    file_reader = parse_data(data)
    parsed_days = parse_days(file_reader, days)
    output_file = parse_out(out, no_out)

    show_progress_bars = not no_progress and not print_output

    results = []
    for round_num, day_num in parsed_days:
        print(f"Backtesting {algorithm} on round {round_num} day {day_num}")

        reload(trader_module)

        result = run_backtest(
            trader_module.Trader(PARAMS),
            file_reader,
            round_num,
            day_num,
            print_output,
            match_trades,
            True,
            show_progress_bars,
        )

        print_day_summary(result)
        if len(parsed_days) > 1:
            print()

        results.append(result)

    if len(parsed_days) > 1:
        print_overall_summary(results)

    if output_file is not None:
        merged_results = reduce(lambda a, b: merge_results(a, b, merge_pnl, not original_timestamps), results)
        write_output(output_file, merged_results)
        print(f"\nSuccessfully saved backtest results to {format_path(output_file)}")

    if vis and output_file is not None:
        open_visualizer(output_file)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
