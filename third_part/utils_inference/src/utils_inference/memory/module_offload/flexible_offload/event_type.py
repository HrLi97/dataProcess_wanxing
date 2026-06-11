from enum import IntEnum


class EventType(IntEnum):
    HTOD = 0
    HTOH = 1
    DTOH = 2
    DTOD = 3

    # below has no copy type
    CALC = 10

    @staticmethod
    def is_copy(value):
        return value < EventType.CALC
