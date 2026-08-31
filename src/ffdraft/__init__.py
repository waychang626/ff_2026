"""ffdraft - a deterministic draft engine for season-long fantasy football.

The LLM orchestrates; this package decides. `recommend_pick` is the whole
public surface that matters:

    from ffdraft.engine import recommend_pick
    advice = recommend_pick("cuomo", drafted, my_roster, pick_number)
    print(advice.format_card())

It takes no tuning parameters. Risk appetite and position need are derived
inside the engine, from the roster and the live board. See README.md.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
