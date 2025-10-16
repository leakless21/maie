from src.config import configure_logging, get_logger


def main():
    # Always configure Loguru at startup.
    logger = configure_logging()
    logger = logger if logger is not None else get_logger()
    logger.info("Loguru configuration active (phase1) - main")

    # Preserve original behavior
    get_logger().info("Hello from maie!")


if __name__ == "__main__":
    main()
