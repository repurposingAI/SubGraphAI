import logging
from yachalk import chalk
import os
import logging


class GraphLogger:

    def __init__(self, name="Graph Logger", color="white"):
        "Set the log level (optional, can be DEBUG, INFO, WARNING, ERROR, CRITICAL)"

        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=log_level)

        ## Formatter
        self.time_format = "%Y-%m-%d %H:%M:%S"
        format = self.format(color)
        self.formatter = logging.Formatter(
            fmt=format,
            datefmt=self.time_format,
        )

        ## Handler
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)

        ## Logger
        self.logger = logging.getLogger(name)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def getLogger(self):
        return self.logger

    def format(self, color: str):
     if color == "black":
        format = chalk.black(
            "\n▶︎ %(name)s - %(asctime)s - %(levelname)s \n%(message)s\n"
        )
     elif color == "red":
        format = chalk.red(
            "\n▶︎ %(name)s - %(asctime)s - %(levelname)s \n%(message)s\n"
        )
     elif color == "green":
        format = chalk.green(
            "\n▶︎ %(name)s - %(asctime)s - %(levelname)s \n%(message)s\n"
        )
     # Ajoutez d'autres cas elif pour chaque couleur...
     else:
        format = chalk.white(
            "\n▶︎ %(name)s - %(asctime)s - %(levelname)s \n%(message)s\n"
        )

     return format
