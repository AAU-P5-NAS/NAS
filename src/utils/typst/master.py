from src.utils.typst.parser import generate_lilaq_string
import pathlib
import argparse

DIRECTORY = "src/utils/typst/data/"

parser = argparse.ArgumentParser(description="Convert JSON logs to Typst Lilaq plot.")

parser.add_argument("output", type=str, help="Name of output file")
parser.add_argument("--experiments", nargs="+", required=True, help="Name of experiments to include")
parser.add_argument("--y", type=str, required=True, help="Name of metric to plot on y-axis")
parser.add_argument("--resolution", type=int, default=1, help="Downsample factor")
parser.add_argument("--smooth", type=int, default=1, help="Smoothing factor")
parser.add_argument("--xmin", type=float, default=None, help="X minimum")
parser.add_argument("--xmax", type=float, default=None, help="X maximum")
parser.add_argument("--ymin", type=float, default=None, help="Y minimum")
parser.add_argument("--ymax", type=float, default=None, help="Y maximum")

args = parser.parse_args()

lilaq_string = generate_lilaq_string(   
    directory=DIRECTORY,
    experiments=args.experiments,
    metric=args.y,
    resolution=args.resolution,
    smooth_window=args.smooth,
    x_min=args.xmin,
    x_max=args.xmax,
    y_min=args.ymin,
    y_max=args.ymax,
)

with open(pathlib.Path.joinpath(pathlib.Path(DIRECTORY), args.output + ".txt"), "w") as f:
    f.write(lilaq_string)