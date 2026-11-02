"""Pure helpers for the controlled wooden-log interaction curriculum.

No Minecraft/MCio dependencies; safe for unit tests without game downloads.
"""
import re


def validate_tag(tag: str) -> str:
    """Never map a user-supplied run tag outside disposable trial names."""
    if re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}', tag) is None:
        raise ValueError('tag must contain 1-40 letters, digits, underscores or hyphens')
    return f'MDL_LogInputProbe_{tag}'


def inventory_count(observation, item: str = 'minecraft:oak_log') -> int:
    """Count only *observed* real inventory items; never infer from commands."""
    return sum(slot.count for slot in observation.inventory_main
               if slot.id == item and slot.count > 0)


def relative_cursor(observation, dx_px: float, dy_px: float):
    """Convert an agent's pixel delta to MCio's absolute cursor position."""
    return [(float(observation.cursor_pos[0]) + dx_px,
             float(observation.cursor_pos[1]) + dy_px)]
