from services.jobs import JobTracker


def test_create_returns_unique_ids() -> None:
    t = JobTracker()
    ids = {t.create("x") for _ in range(50)}
    assert len(ids) == 50


def test_initial_state() -> None:
    t = JobTracker()
    jid = t.create("scan")
    job = t.get(jid)
    assert job["name"] == "scan"
    assert job["status"] == "running"
    assert job["progress"] == 0
    assert job["total"] == 0
    assert job["result"] is None


def test_update_progress() -> None:
    t = JobTracker()
    jid = t.create("scan")
    t.update(jid, 3, 10)
    assert t.get(jid)["progress"] == 3
    assert t.get(jid)["total"] == 10


def test_complete_sets_result_and_status() -> None:
    t = JobTracker()
    jid = t.create("scan")
    t.complete(jid, {"imported": 42})
    job = t.get(jid)
    assert job["status"] == "completed"
    assert job["result"] == {"imported": 42}


def test_fail_sets_error_status() -> None:
    t = JobTracker()
    jid = t.create("scan")
    t.fail(jid, "boom")
    job = t.get(jid)
    assert job["status"] == "failed"
    assert job["result"] == {"error": "boom"}


def test_get_returns_none_for_unknown() -> None:
    t = JobTracker()
    assert t.get("not-a-real-id") is None


def test_update_unknown_job_is_noop() -> None:
    t = JobTracker()
    t.update("nope", 1, 2)  # should not raise
    assert t.get("nope") is None


def test_list_all_returns_all_jobs() -> None:
    t = JobTracker()
    a = t.create("a")
    b = t.create("b")
    all_jobs = t.list_all()
    assert a in all_jobs and b in all_jobs
    assert all_jobs[a]["name"] == "a"
