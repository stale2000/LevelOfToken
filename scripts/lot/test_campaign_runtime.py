"""Focused tests for pure campaign deadline and Fal admission logic."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from campaign_runtime import (
    CampaignDeadlines,
    FalLimits,
    WallDeadline,
    admit_request,
    make_monotonic_deadline,
)

UTC = timezone.utc
START = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class DeadlineTests(unittest.TestCase):
    def test_wall_deadlines_are_utc_and_have_exact_boundaries(self):
        pacific = timezone(timedelta(hours=-7))
        schedule = CampaignDeadlines.from_start(START.astimezone(pacific))
        self.assertEqual(schedule.started_at, START)
        self.assertEqual(schedule.optimizer_cutoff.at, START + timedelta(hours=4))
        self.assertEqual(schedule.last_submission.at, START + timedelta(hours=4, minutes=30))
        self.assertEqual(schedule.final_deadline.at, START + timedelta(hours=6))
        deadline = WallDeadline(START + timedelta(seconds=10))
        self.assertFalse(deadline.expired(START + timedelta(seconds=9)))
        self.assertTrue(deadline.expired(START + timedelta(seconds=10)))

    def test_wall_deadline_rejects_naive_datetime(self):
        with self.assertRaisesRegex(ValueError, "include a timezone"):
            CampaignDeadlines.from_start(datetime(2026, 10, 10, 12))  # noqa: DTZ001

    def test_monotonic_deadline_does_not_depend_on_wall_clock(self):
        deadline = make_monotonic_deadline(30, now_monotonic=1000)
        self.assertEqual(deadline.at, 1030)
        self.assertEqual(deadline.remaining_seconds(1010), 20)
        self.assertFalse(deadline.expired(1029.9))
        self.assertTrue(deadline.expired(1030))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.schedule = CampaignDeadlines.from_start(START)

    def test_fal_limit_boundaries_are_admitted(self):
        result = admit_request(
            now_utc=START, deadlines=self.schedule, optimizer_seconds=0,
            evaluation_seconds=900, export_reserve_seconds=900,
        )
        self.assertTrue(result.admitted, result.reason)
        self.assertEqual(result.processing_seconds, 1800)
        self.assertEqual(result.reason, "fits campaign and Fal request deadlines")

    def test_eval_and_full_export_reserve_share_45_minute_request_budget(self):
        fits = admit_request(
            now_utc=START, deadlines=self.schedule, optimizer_seconds=900,
            evaluation_seconds=900, export_reserve_seconds=900,
        )
        self.assertTrue(fits.admitted, fits.reason)
        too_long = admit_request(
            now_utc=START, deadlines=self.schedule, optimizer_seconds=901,
            evaluation_seconds=900, export_reserve_seconds=900,
        )
        self.assertFalse(too_long.admitted)
        self.assertIn("includes evaluation and export", too_long.reason)

    def test_final_deadline_reserves_startup_grace_and_safety_margin(self):
        limits = FalLimits()
        # Tight synthetic campaign isolates the final-deadline guard from the
        # real schedule's earlier last-submission cutoff.
        tight_schedule = replace(
            self.schedule,
            last_submission=WallDeadline(START + timedelta(hours=6)),
            final_deadline=WallDeadline(START + timedelta(seconds=2400)),
        )
        at_boundary = admit_request(
            now_utc=START, deadlines=tight_schedule, phase="evaluation",
            optimizer_seconds=0, evaluation_seconds=120,
            export_reserve_seconds=0, limits=limits,
        )
        self.assertTrue(at_boundary.admitted, at_boundary.reason)
        after_boundary = admit_request(
            now_utc=START, deadlines=replace(
                tight_schedule,
                final_deadline=WallDeadline(START + timedelta(seconds=2399)),
            ),
            phase="evaluation", optimizer_seconds=0, evaluation_seconds=120,
            export_reserve_seconds=0, limits=limits,
        )
        self.assertFalse(after_boundary.admitted)
        self.assertIn("T+6h final deadline", after_boundary.reason)

    def test_optimizer_and_submission_cutoffs(self):
        no_startup = FalLimits(server_start_timeout=0, startup_timeout=0)
        exact_cutoff = admit_request(
            now_utc=self.schedule.optimizer_cutoff.at, deadlines=self.schedule,
            optimizer_seconds=0, export_reserve_seconds=0, limits=no_startup,
        )
        self.assertTrue(exact_cutoff.admitted, exact_cutoff.reason)
        after_cutoff = admit_request(
            now_utc=self.schedule.optimizer_cutoff.at + timedelta(seconds=1),
            deadlines=self.schedule, optimizer_seconds=0, export_reserve_seconds=0,
            limits=no_startup,
        )
        self.assertFalse(after_cutoff.admitted)
        self.assertIn("cannot submit after optimizer cutoff", after_cutoff.reason)

        queued_past_cutoff = admit_request(
            now_utc=self.schedule.optimizer_cutoff.at - timedelta(seconds=100),
            deadlines=self.schedule, optimizer_seconds=0, export_reserve_seconds=0,
            limits=FalLimits(server_start_timeout=90, startup_timeout=20),
        )
        self.assertFalse(queued_past_cutoff.admitted)
        self.assertIn("optimizer cutoff", queued_past_cutoff.reason)

        after_last_submit = admit_request(
            now_utc=self.schedule.last_submission.at + timedelta(seconds=1),
            deadlines=self.schedule, phase="evaluation", optimizer_seconds=0,
            export_reserve_seconds=0,
        )
        self.assertFalse(after_last_submit.admitted)
        self.assertIn("last submission cutoff", after_last_submit.reason)

    def test_useful_configuration_failure_reasons(self):
        cases = (
            (FalLimits(server_start_timeout=901), "server start timeout"),
            (FalLimits(startup_timeout=901), "startup timeout"),
            (FalLimits(request_timeout=2701), "request timeout"),
            (FalLimits(termination_grace=181), "termination grace"),
            (FalLimits(safety_margin=299), "safety margin"),
        )
        for limits, message in cases:
            with self.subTest(message=message):
                result = admit_request(
                    now_utc=START, deadlines=self.schedule, optimizer_seconds=0,
                    export_reserve_seconds=0, limits=limits,
                )
                self.assertFalse(result.admitted)
                self.assertIn(message, result.reason)


if __name__ == "__main__":
    unittest.main()
