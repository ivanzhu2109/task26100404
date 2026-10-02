"""Tests for the GPU gang scheduler."""

import unittest

from gpu_scheduler import GpuScheduler


class SchedulerTest(unittest.TestCase):
    def test_two_single_gpu_jobs_both_fit(self) -> None:
        # Front job compatible with both cards, rear job with only one:
        # both must be selected.
        scheduler = GpuScheduler()
        scheduler.register("front", 1, 1, {0, 1})
        scheduler.register("rear", 1, 1, {1})
        plan = scheduler.plan({0, 1})
        self.assertEqual(set(plan.assignments), {"front", "rear"})
        self.assertEqual(plan.gpu_ids_for("rear"), frozenset({1}))
        self.assertEqual(plan.gpu_ids_for("front"), frozenset({0}))

    def test_double_gpu_job_beats_two_single_gpu_jobs(self) -> None:
        # High-priority 2-GPU job conflicts with two low-priority 1-GPU
        # jobs on 2 cards: the former wins, and is never split.
        scheduler = GpuScheduler()
        scheduler.register("big", 1, 2, {0, 1})
        scheduler.register("small-1", 2, 1, {0, 1})
        scheduler.register("small-2", 2, 1, {0, 1})
        plan = scheduler.plan({0, 1})
        self.assertEqual(set(plan.assignments), {"big"})
        self.assertEqual(plan.gpu_ids_for("big"), frozenset({0, 1}))

    def test_same_priority_uses_registration_order(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register("first", 2, 1, {0})
        scheduler.register("second", 2, 1, {0})
        plan = scheduler.plan({0})
        self.assertEqual(set(plan.assignments), {"first"})

    def test_unfittable_job_is_skipped_not_blocking(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register("greedy", 1, 3, {0, 1, 2})
        scheduler.register("next", 1, 1, {0})
        plan = scheduler.plan({0, 1})
        self.assertEqual(set(plan.assignments), {"next"})

    def test_no_partial_allocation(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register("duo", 1, 2, {0, 1})
        plan = scheduler.plan({0})
        self.assertEqual(plan.assignments, {})

    def test_retains_previous_pairings(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register("a", 1, 1, {0, 1})
        scheduler.register("b", 1, 1, {0, 1})
        first = scheduler.plan({0, 1})
        # Replan with identical inputs: assignment must be identical.
        second = scheduler.plan({0, 1})
        self.assertEqual(first.assignments, second.assignments)

    def test_retention_is_best_effort(self) -> None:
        # Job "a" had gpus {0, 1}; after gpu 1 disappears it keeps gpu 0.
        scheduler = GpuScheduler()
        scheduler.register("a", 1, 2, {0, 1, 2})
        scheduler.plan({0, 1, 2})
        scheduler.register("b", 1, 1, {2})
        plan = scheduler.plan({0, 2})
        self.assertEqual(plan.gpu_ids_for("a"), frozenset({0, 2}))
        self.assertEqual(plan.assignments.get("b"), None)

    def test_retention_never_breaks_queue_order(self) -> None:
        # Keeping an old pairing for the rear job must not evict the front
        # job even though both want the same card.
        scheduler = GpuScheduler()
        scheduler.register("front", 1, 1, {0, 1})
        scheduler.register("rear", 2, 1, {0, 1})
        scheduler.plan({0, 1})
        plan = scheduler.plan({0})
        self.assertEqual(set(plan.assignments), {"front"})

    def test_withdraw_releases_resources(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register("a", 1, 1, {0})
        scheduler.register("b", 1, 1, {0})
        scheduler.plan({0})
        scheduler.withdraw("a")
        plan = scheduler.plan({0})
        self.assertEqual(set(plan.assignments), {"b"})

    def test_unavailable_gpu_frees_old_assignment(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register("a", 1, 1, {0})
        scheduler.register("b", 1, 1, {0, 1})
        scheduler.plan({0, 1})
        # gpu 0 becomes unavailable; "a" drops out, "b" can take gpu 1.
        plan = scheduler.plan({1})
        self.assertEqual(set(plan.assignments), {"b"})
        self.assertEqual(plan.gpu_ids_for("b"), frozenset({1}))

    def test_validation(self) -> None:
        scheduler = GpuScheduler()
        with self.assertRaises(ValueError):
            scheduler.register("bad-prio", 0, 1, {0})
        with self.assertRaises(ValueError):
            scheduler.register("bad-prio-2", 6, 1, {0})
        with self.assertRaises(ValueError):
            scheduler.register("bad-count", 1, 0, {0})
        with self.assertRaises(ValueError):
            scheduler.register("bad-compat", 1, 2, {0})


if __name__ == "__main__":
    unittest.main()
