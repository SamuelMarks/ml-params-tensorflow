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
    if len(args) == 1:
        return args[0]
    return args
