"""Simple dead code detector wrapper using `deadcode` library."""

import sys
from deadcode.__main__ import main as deadcode_main


def main() -> int:
    args = ["deadcode", "src", "tests", "--exit-zero-unused-ignore" ]
    try:
        return deadcode_main(args)
    except SystemExit as exc:  # deadcode may sys.exit
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(main())



