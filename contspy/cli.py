import sys

from .__about__ import __version__


def show(argv=None):
    # Parse command line arguments.
    parser = _get_parser()
    args = parser.parse_args(argv)

    print(args.number)
    return


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=("Dummy contspy executable."),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("number", type=int, help="number to show")

    __copyright__ = "Copyright (c) 2020 Antoine Jacquey <ajacquey@mit.edu>"
    version_text = "\n".join(
        [
            "contspy {} [Python {}.{}.{}]".format(
                __version__,
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ),
            __copyright__,
        ]
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=version_text,
        help="display version information",
    )

    return parser
