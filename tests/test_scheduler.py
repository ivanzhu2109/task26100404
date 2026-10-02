import pytest

from gpu_scheduler import Job, Scheduler


def test_front_job_compatible_with_both_cards_both_selected():
    s = Scheduler()
    s.register(Job("front", 1, 1, frozenset({"g0", "g1"})))
    s.register(Job("back", 2, 1, frozenset({"g1"})))
    plan = s.plan(frozenset({"g0", "g1"}))
    assert set(plan.assignments) == {"front", "back"}
    assert plan.assignments["back"] == frozenset({"g1"})
    assert plan.assignments["front"] == frozenset({"g0"})


def test_high_priority_two_gpu_job_beats_two_single_gpu_jobs():
    s = Scheduler()
    s.register(Job("big", 1, 2, frozenset({"g0", "g1"})))
    s.register(Job("s1", 2, 1, frozenset({"g0"})))
    s.register(Job("s2", 2, 1, frozenset({"g1"})))
    plan = s.plan(frozenset({"g0", "g1"}))
    assert plan.assignments == {"big": frozenset({"g0", "g1"})}


def test_same_priority_ordered_by_registration():
    s = Scheduler()
    s.register(Job("first", 1, 1, frozenset({"g0"})))
    s.register(Job("second", 1, 1, frozenset({"g0"})))
    plan = s.plan(frozenset({"g0"}))
    assert plan.assignments == {"first": frozenset({"g0"})}


def test_infeasible_job_skipped_later_job_still_considered():
    s = Scheduler()
    s.register(Job("too_big", 1, 3, frozenset({"g0", "g1", "g2"})))
    s.register(Job("small", 2, 1, frozenset({"g0"})))
    plan = s.plan(frozenset({"g0", "g1"}))
    assert plan.assignments == {"small": frozenset({"g0"})}


def test_partial_allocation_never_happens():
    s = Scheduler()
    s.register(Job("need_two", 1, 2, frozenset({"g0", "g1", "g2"})))
    plan = s.plan(frozenset({"g0"}))
    assert plan.assignments == {}


def test_previous_pairing_preserved_when_possible():
    s = Scheduler()
    s.register(Job("a", 1, 2, frozenset({"g0", "g1", "g2", "g3"})))
    plan1 = s.plan(frozenset({"g0", "g1"}))
    assert plan1.assignments["a"] == frozenset({"g0", "g1"})
    # 可用集合扩大后重排，旧配对应全部保留
    plan2 = s.plan(frozenset({"g0", "g1", "g2", "g3"}))
    assert plan2.assignments["a"] == frozenset({"g0", "g1"})


def test_unavailable_gpu_frees_old_pairing():
    s = Scheduler()
    s.register(Job("a", 1, 2, frozenset({"g0", "g1", "g2"})))
    plan1 = s.plan(frozenset({"g0", "g1"}))
    assert plan1.assignments["a"] == frozenset({"g0", "g1"})
    # g1 不再可用，a 需重分到 g0+g2，保留 g0 这一个配对
    plan2 = s.plan(frozenset({"g0", "g2"}))
    assert plan2.assignments["a"] == frozenset({"g0", "g2"})


def test_withdraw_removes_job_and_its_old_assignment():
    s = Scheduler()
    s.register(Job("a", 1, 1, frozenset({"g0"})))
    s.register(Job("b", 2, 1, frozenset({"g0"})))
    assert set(s.plan(frozenset({"g0"})).assignments) == {"a"}
    s.withdraw("a")
    plan = s.plan(frozenset({"g0"}))
    assert plan.assignments == {"b": frozenset({"g0"})}
    with pytest.raises(KeyError):
        s.withdraw("a")


def test_duplicate_registration_rejected():
    s = Scheduler()
    s.register(Job("a", 1, 1, frozenset({"g0"})))
    with pytest.raises(ValueError):
        s.register(Job("a", 2, 1, frozenset({"g1"})))


def test_invalid_job_attributes_rejected():
    with pytest.raises(ValueError):
        Job("x", 0, 1, frozenset({"g0"}))
    with pytest.raises(ValueError):
        Job("x", 1, 0, frozenset({"g0"}))
    with pytest.raises(ValueError):
        Job("x", 1, 1, frozenset())


def test_plan_output_becomes_next_previous():
    s = Scheduler()
    s.register(Job("a", 1, 1, frozenset({"g0", "g1"})))
    s.register(Job("b", 2, 1, frozenset({"g0", "g1"})))
    plan1 = s.plan(frozenset({"g0", "g1"}))
    plan2 = s.plan(frozenset({"g0", "g1"}))
    # 无变化时计划应保持稳定
    assert plan2.assignments == plan1.assignments
