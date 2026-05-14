"""Batch and high-quality rendering utilities (manifest-based 2D/3D, IAGNet/Mitsuba)."""


def render_targets_from_json(*args, **kwargs):
    """Lazy wrapper to avoid importing batch_render when running it as a module."""
    from utils.rendering.batch_render import render_targets_from_json as _impl

    return _impl(*args, **kwargs)


__all__ = ["render_targets_from_json"]
