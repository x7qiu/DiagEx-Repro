"""Local browser workbench for configuring and running DiagEx."""

__all__ = ["Workbench", "serve_workbench"]


def __getattr__(name):
    # The CV worker runs in a Torch environment without the web/LLM dependencies.
    if name in __all__:
        from diagex.web import server

        return getattr(server, name)
    raise AttributeError(name)
