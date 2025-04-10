import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated, Optional

from typer import Argument, Option, Typer

from prosperity3submit.submit import submit

import re

def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity3submit {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm to submit.", show_default=False, exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    
    
    reversion_beta: Annotated[Optional[float], Option("--reversion_beta", help="Custom reversion beta value.")] = -0.529,
    take_width: Annotated[Optional[float], Option("--take_width", help="Custom take width value.")] = 1,
    clear_width: Annotated[Optional[float], Option("--clear_width", help="Custom clear width value.")] = 0,
    adverse_volume: Annotated[Optional[int], Option("--adverse_volume", help="Custom adverse volume.")] = 15,
    disregard_edge: Annotated[Optional[float], Option("--disregard_edge", help="Custom adverse volume.")] = 15,
    join_edge: Annotated[Optional[float], Option("--join_edge", help="Custom adverse volume.")] = 15,
    default_edge: Annotated[Optional[float], Option("--default_edge", help="Custom adverse volume.")] = 15,
    z_rolling_window: Annotated[Optional[int], Option("--z_rolling_window", help="Custom adverse volume.")] = 15,
    zscore_threshold: Annotated[Optional[float], Option("--zscore_threshold", help="Custom adverse volume.")] = 15,

    out: Annotated[Optional[str], Option(help="File to save submission logs to (defaults to submissions/<timestamp>.log).", show_default=False, dir_okay=False, resolve_path=True)] = None,
    no_out: Annotated[bool, Option("--no-out", help="Don't download logs when done.")] = False,
    vis: Annotated[bool, Option("--vis", help="Open backtest results in https://jmerle.github.io/imc-prosperity-3-visualizer/ when done.")] = False,
    version: Annotated[bool, Option("--version", "-v", help="Show the program's version number and exit.", is_eager=True, callback=version_callback)] = False,
) -> None:  # fmt: skip
    
    # Read the template
    with open("template.py", 'r') as file:
        content = file.read()

    # Update only the Product.SQUID_INK parameters without using groups
    content = re.sub(r'take_width_to_be_changed',str(take_width), content)
    content = re.sub(r'clear_width_to_be_changed',str(clear_width), content)
    content = re.sub(r'adverse_volume_to_be_changed',str(adverse_volume), content)
    content = re.sub(r'reversion_beta_to_be_changed',str(reversion_beta), content)
    content = re.sub(r'disregard_edge_to_be_changed',str(disregard_edge), content)
    content = re.sub(r'join_edge_to_be_changed',str(join_edge), content)
    content = re.sub(r'default_edge_to_be_changed',str(default_edge), content)
    content = re.sub(r'z_rolling_window_to_be_changed',str(z_rolling_window), content)
    content = re.sub(r'zscore_threshold_to_be_changed',str(zscore_threshold), content)

    # Change parameters
    with open(algorithm, 'w') as file:
        file.write(content)


    """Submit an IMC Prosperity 3 algorithm."""
    if out is not None and no_out:
        print("--out and --no-logs are mutually exclusive")
        sys.exit(1)

    if no_out and vis:
        print("--no-logs and --vis are mutually exclusive")
        sys.exit(1)

    if out is not None:
        output_file = Path(out).expanduser().resolve()
    elif no_out:
        output_file = None
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = Path.cwd() / "submissions" / f"{timestamp}.log"



    submit(algorithm, output_file, vis)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
