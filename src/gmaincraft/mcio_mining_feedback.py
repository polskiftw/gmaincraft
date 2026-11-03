"""Opt-in, read-only vanilla Minecraft block-breaking animation feedback.

This observation is PLAYER-OWNED / HUD-visible, NOT an evaluator reward or
hidden world map. The game client already knows whether this player is
breaking a block and how far its visible crack animation has progressed.
Register the CBOR type BEFORE the first MCio observation is decoded.
"""
from dataclasses import dataclass

from mcio_ctrl.cbor import MCioType

SCHEMA = 'gmaincraft_mining_feedback_v1'
ENV_NAME = 'GMAINCRAFT_MINING_FEEDBACK_V1'


@MCioType
@dataclass
class MiningFeedbackOption:
    schema: str
    actively_destroying: bool
    crack_stage: int
    crosshair_on_block: bool


@dataclass(frozen=True)
class MiningFeedbackV1:
    schema: str
    actively_destroying: bool
    crack_stage: int
    crosshair_on_block: bool

    @classmethod
    def from_packet(cls, packet):
        option = packet.get_option(MiningFeedbackOption)
        if option is None:
            raise RuntimeError('Mining feedback absent: patched MCio mod and '
                               'GMAINCRAFT_MINING_FEEDBACK_V1=1 required')
        if option.schema != SCHEMA:
            raise ValueError('Unexpected mining feedback schema')
        active, stage, targeted = (option.actively_destroying,
                                   option.crack_stage, option.crosshair_on_block)
        if type(active) is not bool or type(targeted) is not bool or type(stage) is not int:
            raise ValueError('Invalid mining feedback types')
        if active and not 0 <= stage <= 10:
            raise ValueError('Active mining crack stage out of range')
        if not active and stage != -1:
            raise ValueError('Inactive mining must report unknown crack stage')
        return cls(SCHEMA,active,stage,targeted)

    @property
    def normalized_crack(self):
        """A bounded visible animation fraction; -1 means no active break.

        The actual Java stage is integer stepped. 1.0 is a visible crack
        stage, NOT a claim that the block has been successfully mined.
        """
        return self.crack_stage / 10.0 if self.actively_destroying else -1.0
