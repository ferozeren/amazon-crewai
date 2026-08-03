import sys
from pathlib import Path


def main() -> None:
    sys_info = system_info()
    print(sys_info)


def system_info() -> dict[str, str]:

    return {"py_version": sys.version, "pwd": str(Path.cwd())}


if __name__ == "__main__":
    main()
