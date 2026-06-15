"""No-logger entry point for the Black-Scholes calculator."""

from Model import main as model_main


def main():
    """Run the calculator without enabling the data logger."""
    return model_main(enable_logging=False)


if __name__ == "__main__":
    main()