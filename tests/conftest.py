import pytest

import multiprocessing as mp


# cannot parametrize unittest.TestCase. We should test both
# fork and spawn but I'm not sure how to.
# @pytest.fixture(params=["fork", "spawn"], autouse=True)
@pytest.fixture(autouse=True)
def context(monkeypatch):
    ctx = mp.get_context("spawn")
    monkeypatch.setattr(mp, "Queue", ctx.Queue)
    monkeypatch.setattr(mp, "Process", ctx.Process)
    monkeypatch.setattr(mp, "Event", ctx.Event)
    monkeypatch.setattr(mp, "Value", ctx.Value)
