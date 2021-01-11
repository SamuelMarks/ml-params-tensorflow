"""
Utility functions
"""


def identity(*args):
    """
    Identity function

    :param args: Splat of arguments
    :type args: ```tuple```

    :return: First arg if len(args) is 1 else all args
    :rtype: Union[Tuple[Any, Tuple[Any]]]
    """
    return args[0] if len(args) == 1 else args
