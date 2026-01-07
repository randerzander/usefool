import sys
import logging
from loguru import logger
from rich.console import Console
from rich.text import Text

# Shared console instance with forced terminal colors and interactivity
# force_interactive=True ensures spinners update in-place even if TTY detection fails
console = Console(force_terminal=True, force_interactive=True)

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame = logging.currentframe()
        depth = 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__ or "logging" in frame.f_code.co_filename):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def rich_sink(message):
    """
    Custom sink that converts ANSI-colored loguru messages to Rich Text objects
    and prints them to the shared console. This preserves the spinner state.
    """
    # Remove the trailing newline since console.print adds one, 
    # but loguru message includes one.
    msg_str = message.rstrip("\n")
    
    # Convert ANSI codes (from loguru) to Rich Text
    text = Text.from_ansi(msg_str)
    
    # Print to console (handles spinner safely)
    console.print(text)

def setup_logging():
    """Set up Loguru logger and intercept standard logging using Rich."""
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Remove default loguru handler
    logger.remove()
    
    # Add custom Rich sink
    logger.add(
        rich_sink,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True, # Loguru generates ANSI codes, we parse them in sink
    )
    
    # Suppress noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Also provide a way to get the shared console
    return console