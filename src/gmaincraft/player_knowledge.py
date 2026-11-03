"""Version 1 bounded, read-only player-knowledge observations from MCio.

This is a *new* opt-in contract, not a silent change to legacy RGB AgentView.
No hidden world map, target/block position, command, or raw MCio packet leaks.
The same constructor is intended for both training and frozen evaluation.
"""
from dataclasses import dataclass
from math import isfinite


SCHEMA = 'player_knowledge_v1'
# Only player-owned counters already maintained by MCio are exposed. These
# must be available in the underlying stats cache; unknown != zero.
PLAYER_STAT_CATEGORIES = frozenset(('minecraft:mined', 'minecraft:picked_up',
                                    'minecraft:used', 'minecraft:crafted',
                                    'minecraft:broken', 'minecraft:dropped'))
MAX_INVENTORY_SLOTS = 46
MAX_COUNTERS = 512


@dataclass(frozen=True)
class OwnedItem:
    slot: int
    item_id: str
    count: int


@dataclass(frozen=True)
class PlayerKnowledgeV1:
    """Player's own knowable state; no direct visibility/block-list claim.

    Unknown stats remain absent. StatsCache must be updated each game tick
    by its owning bridge, not inferred from differences between episodes.
    """
    schema: str
    player_xyz: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    health: float
    main_inventory: tuple[OwnedItem, ...]
    armor_inventory: tuple[OwnedItem, ...]
    offhand_inventory: tuple[OwnedItem, ...]
    progress_stats: tuple[tuple[str, str, int], ...]

    @classmethod
    def from_packet(cls, packet, *, stats=None):
        def finite_number(value):
            x = float(value)
            if not isfinite(x):
                raise ValueError('Non-finite player state')
            return x

        def inventory(slots):
            if len(slots) > MAX_INVENTORY_SLOTS:
                raise ValueError('Unexpected oversized player inventory')
            result = []
            for item in slots:
                slot, count, item_id = int(item.slot), int(item.count), str(item.id)
                if not 0 <= slot <= 255 or not 0 <= count <= 2304:
                    raise ValueError('Invalid player-owned inventory slot/count')
                if count > 0:
                    if not item_id.startswith('minecraft:'):
                        # Other mod namespaces are ordinary item IDs; don't
                        # forbid their existence, just keep a bounded value.
                        if ':' not in item_id:
                            raise ValueError('Invalid item identifier')
                    if len(item_id) > 128:
                        raise ValueError('Item identifier too long')
                    result.append(OwnedItem(slot, item_id, count))
            return tuple(sorted(result, key=lambda item: item.slot))

        counters = []
        if stats is not None:
            for category in sorted(PLAYER_STAT_CATEGORIES):
                for item_id, value in sorted(stats.get(category, {}).items()):
                    value = int(value)
                    if value < 0 or len(str(item_id)) > 128:
                        raise ValueError('Invalid player statistic')
                    counters.append((category, str(item_id), value))
            if len(counters) > MAX_COUNTERS:
                raise ValueError('Player statistic budget exceeded')
        xyz = tuple(finite_number(v) for v in packet.player_pos)
        if len(xyz) != 3:
            raise ValueError('Expected XYZ player position')
        return cls(SCHEMA, xyz, finite_number(packet.player_yaw),
                   finite_number(packet.player_pitch), finite_number(packet.health),
                   inventory(packet.inventory_main),inventory(packet.inventory_armor),
                   inventory(packet.inventory_offhand),tuple(counters))
