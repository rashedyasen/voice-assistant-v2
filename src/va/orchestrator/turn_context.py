import multiprocessing
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event


@dataclass
class TurnContext:
    turn_id: int
    cancelled: Event = field(default_factory=multiprocessing.Event)
