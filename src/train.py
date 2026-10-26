import argparse

from src.pipeline import generate_dataset, train_and_report


def main() -> None:
    # Expose data generation and training through one repeatable command.
    parser = argparse.ArgumentParser(description="Train the visual inspection models.")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--inject-leak", action="store_true")
    parser.add_argument("--lighting-shift", type=float, default=0.0)
    args = parser.parse_args()

    generate_dataset(
        inject_leak=args.inject_leak,
        lighting_shift=args.lighting_shift,
    )
    if not args.data_only:
        train_and_report()


if __name__ == "__main__":
    main()