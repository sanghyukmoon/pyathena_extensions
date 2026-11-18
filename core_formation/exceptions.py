class NoNearbyCoreError(Exception):
    pass


class TCOLLcoreContainsSinkError(ValueError):
    """The selected minimum immediately before collapse contains a sink."""
