import logging


def setup_logging(logger=None,
                  log_file=None,
                  log_level=logging.INFO,
                  format='%(asctime)s - %(levelname)s - %(message)s',
                  datefmt='Y-%m-%d %H:%M:%S'):
    if not logger:
        logger = logging.getLogger()

    logging.basicConfig(level=log_level,
                        format=format,
                        datefmt=datefmt,
                        filename=str(log_file))
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
