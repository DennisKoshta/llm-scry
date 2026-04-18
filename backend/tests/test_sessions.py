from llm_scry.sessions import SessionStore


def test_lru_eviction():
    store = SessionStore(capacity=2)
    a = store.new_session("gpt2", "hello")
    b = store.new_session("gpt2", "world")
    c = store.new_session("gpt2", "third")
    assert store.get(a.id) is None
    assert store.get(b.id) is not None
    assert store.get(c.id) is not None


def test_get_touches_recency():
    store = SessionStore(capacity=2)
    a = store.new_session("gpt2", "a")
    b = store.new_session("gpt2", "b")
    # Touching `a` makes it most-recent; next insertion should evict `b`.
    store.get(a.id)
    store.new_session("gpt2", "c")
    assert store.get(a.id) is not None
    assert store.get(b.id) is None
