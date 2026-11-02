"""Launch settings from the published RTX 5060 Ti full-step benchmarks."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    storage: str
    tiles: int
    warps: int
    direct_cpu: bool
    short_mma_warps: int = 0


PROFILES = {
    p.name: p for p in (
        Profile("balanced", "half16_v8", 8, 8, True, 8),
        Profile("short128", "half16_v8b128", 8, 4, True, 4),
        Profile("conservative", "split24_v8", 2, 8, False, 8),
        Profile("same_as_v5", "tile32", 2, 4, False),
        Profile("experimental_m4v4", "half16_m4v4_sqrt", 1, 4, True),
        Profile("experimental_fp8_master", "float8_m4v4_sqrt", 2, 4, True),
    )
}


def list_profiles():
    return tuple(PROFILES)


def profile_config(name):
    try:
        return PROFILES[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown profile {name!r}. Choose one of {', '.join(PROFILES)}") from None
