"""
Revenium Middleware — the economic system of record for AI usage.

This package provides core metering functionality and provider-specific
middleware for deeply attributed AI usage metrics.
"""
import os
import sys
import logging
import re


class ReadableFormatter(logging.Formatter):
    """
    Custom formatter that improves readability of log messages.
    - Adds visual separators for important events
    - Uses colors if terminal supports them
    - Keeps messages clean and scannable
    """

    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'RED': '\033[91m',
    }

    # Box drawing characters
    SEPARATOR = "─" * 70

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = self._supports_color()

    def _supports_color(self):
        """Check if terminal supports ANSI colors."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _colorize(self, text, color):
        """Apply color to text if terminal supports it."""
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['RESET']}"
        return text

    def _truncate_long_objects(self, message, max_length=200):
        """Truncate very long object representations to keep logs readable."""
        # Check if message contains object representations
        if '(' in message and ')' in message and len(message) > max_length:
            # Look for patterns like "ClassName(field=value, ...)"
            # Match object representations
            pattern = r'(\w+)\(([^)]{100,})\)'

            def replacer(match):
                class_name = match.group(1)
                content = match.group(2)
                # Truncate the content
                if len(content) > 150:
                    truncated = content[:150] + "..."
                    return f"{class_name}({truncated})"
                return match.group(0)

            message = re.sub(pattern, replacer, message)

        return message

    def format(self, record):
        # Format the base message
        message = record.getMessage()

        # Truncate very long object representations in DEBUG logs
        if record.levelname == 'DEBUG':
            message = self._truncate_long_objects(message)

        # Add visual enhancements for key events
        # Check log level first to ensure it takes precedence
        if record.levelname == 'ERROR':
            message = self._colorize(f"[ERROR] {message}", 'RED')
        elif record.levelname == 'WARNING':
            message = self._colorize(f"[WARNING] {message}", 'YELLOW')
        elif "SUCCESS" in message or "successful" in message.lower():
            message = self._colorize(f"[SUCCESS] {message}", 'GREEN')
        elif "FAILURE" in message:
            message = self._colorize(f"[ERROR] {message}", 'RED')
        elif "Shutdown complete" in message:
            # Make shutdown completion more visible
            separator = self._colorize(self.SEPARATOR, 'GREEN')
            success_msg = self._colorize(
                "[COMPLETE] All operations finished successfully", 'GREEN'
            )
            message = f"\n{separator}\n{success_msg}\n{separator}"
        elif "Shutdown initiated" in message:
            separator = self._colorize(self.SEPARATOR, 'CYAN')
            shutdown_msg = self._colorize('[SHUTDOWN] ' + message, 'CYAN')
            message = f"\n{separator}\n{shutdown_msg}"

        # Format timestamp and level
        if record.levelname == 'DEBUG':
            level = self._colorize('DEBUG', 'DIM')
        elif record.levelname == 'INFO':
            level = self._colorize('INFO', 'BLUE')
        elif record.levelname == 'WARNING':
            level = self._colorize('WARN', 'YELLOW')
        elif record.levelname == 'ERROR':
            level = self._colorize('ERROR', 'RED')
        else:
            level = record.levelname

        # Build the final log message
        timestamp = self.formatTime(record, '%H:%M:%S')
        return f"{timestamp} [{level}] {message}"


# Set up logger
logger = logging.getLogger("revenium_middleware")
log_level = os.environ.get("REVENIUM_LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, log_level))
except AttributeError:
    logger.setLevel(logging.INFO)
    logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")

# Configure a handler with the readable formatter if none exists
if not logger.handlers and not logging.root.handlers:
    handler = logging.StreamHandler()
    formatter = ReadableFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Allow propagation to root logger for testing
logger.propagate = True

# Re-export everything from _core for backward compatibility.
# Existing imports like `from revenium_middleware import client` keep working.
from ._core import (  # noqa: E402
    client,
    run_async_in_thread,
    shutdown_event,
    revenium_meter,
    revenium_metadata,
    track_usage,
    is_inside_decorated_function,
    get_function_metadata,
    set_decorated_context,
    clear_decorated_context,
    get_injected_metadata,
    set_injected_metadata,
    clear_injected_metadata,
    merge_metadata,
    idempotency_key,
    get_idempotency_key,
    set_idempotency_key,
    is_selective_metering_enabled,
)

# Re-export tool metering utilities from revenium_metering (v6.8.2+)
from revenium_metering import meter_tool, report_tool_call, configure  # noqa: E402

# Agentic-outcome client (used by examples/agentic_outcomes/ pack)
from .agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings  # noqa: E402

__all__ = [
    # Metering exports
    "client",
    "run_async_in_thread",
    "shutdown_event",
    # Decorator exports
    "revenium_meter",
    "revenium_metadata",
    "track_usage",
    # Context management exports
    "is_inside_decorated_function",
    "get_function_metadata",
    "set_decorated_context",
    "clear_decorated_context",
    "get_injected_metadata",
    "set_injected_metadata",
    "clear_injected_metadata",
    "merge_metadata",
    "idempotency_key",
    "get_idempotency_key",
    "set_idempotency_key",
    # Config exports
    "is_selective_metering_enabled",
    # Tool metering exports (from revenium_metering)
    "meter_tool",
    "report_tool_call",
    "configure",
    # Agentic-outcome exports
    "AgenticOutcomeClient",
    "AgenticOutcomeSettings",
]
