from __future__ import annotations

from storm_pet.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train-sustain", *__import__("sys").argv[1:]]))
