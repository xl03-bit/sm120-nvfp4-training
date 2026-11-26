"""Name matching shared by frozen projection converters."""
from fnmatch import fnmatchcase


def matches_module(name, patterns):
    """Match a case-sensitive glob against a module path or any named ancestor."""
    paths = [name]
    while "." in name:
        name = name.rsplit(".", 1)[0]
        paths.append(name)
    return any(fnmatchcase(path, pattern) for pattern in patterns for path in paths)
