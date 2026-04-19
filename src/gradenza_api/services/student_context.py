"""
Student context assembly for the analytics/student-insight endpoint.

Each function fetches one slice of student data and returns a compact
markdown string (or empty string if no data).  Only enabled toggles
are fetched.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timezone

from gradenza_api.services.supabase_client import get_service_client

logger = logging.getLogger(__name__)


def _run(fn, *args, **kwargs):
    return asyncio.to_thread(functools.partial(fn, *args, **kwargs))


# ── Identity & subject ─────────────────────────────────────────────────────────

async def fetch_student_identity(class_id: str, student_id: str) -> str:
    """Returns student name, email, and class/subject context."""
    def _query():
        svc = get_service_client()
        # Student user info
        u = (
            svc.table("users")
            .select("display_name, email")
            .eq("id", student_id)
            .maybe_single()
            .execute()
        )
        # Class + subject info
        c = (
            svc.table("classes")
            .select("name, exam_subjects(name, code, exam_systems(name, code))")
            .eq("id", class_id)
            .maybe_single()
            .execute()
        )
        return u.data, c.data

    user_data, class_data = await _run(_query)

    lines = ["## Student Identity"]
    if user_data:
        lines.append(f"- Name: {user_data.get('display_name', 'Unknown')}")
        lines.append(f"- Email: {user_data.get('email', 'Unknown')}")
    if class_data:
        lines.append(f"- Class: {class_data.get('name', 'Unknown')}")
        subj = class_data.get("exam_subjects") or {}
        if subj:
            exam = (subj.get("exam_systems") or {}).get("code", "")
            lines.append(f"- Subject: {subj.get('name', '')} ({exam} {subj.get('code', '')})")
    return "\n".join(lines)


# ── Overall mastery ────────────────────────────────────────────────────────────

async def fetch_overall_mastery(class_id: str, student_id: str) -> str:
    def _query():
        svc = get_service_client()
        rows = (
            svc.table("student_topic_mastery")
            .select("total_marks_awarded, total_marks_available")
            .eq("student_id", student_id)
            .eq("class_id", class_id)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    total_awarded = sum(float(r.get("total_marks_awarded") or 0) for r in rows)
    total_available = sum(float(r.get("total_marks_available") or 0) for r in rows)
    pct = round(total_awarded / total_available * 100, 1) if total_available > 0 else None

    lines = ["## Overall Mastery"]
    lines.append(f"- Total marks awarded: {total_awarded:.1f} / {total_available:.1f}")
    if pct is not None:
        lines.append(f"- Overall score: {pct}%")
    return "\n".join(lines)


# ── Topic mastery ──────────────────────────────────────────────────────────────

async def fetch_topic_mastery(class_id: str, student_id: str) -> str:
    def _query():
        svc = get_service_client()
        rows = (
            svc.table("student_topic_mastery")
            .select("topic, subtopic, attempt_count, total_marks_awarded, total_marks_available, mastery_score, last_practiced_at")
            .eq("student_id", student_id)
            .eq("class_id", class_id)
            .order("mastery_score", desc=False)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    lines = ["## Topic Mastery (weakest first)"]
    for r in rows:
        score = r.get("mastery_score")
        pct = f"{round(float(score) * 100, 1)}%" if score is not None else "n/a"
        awarded = float(r.get("total_marks_awarded") or 0)
        available = float(r.get("total_marks_available") or 0)
        topic = r.get("topic", "Unknown")
        subtopic = r.get("subtopic", "")
        label = f"{topic} — {subtopic}" if subtopic else topic
        lines.append(
            f"- {label}: {pct} ({awarded:.1f}/{available:.1f} marks, {r.get('attempt_count', 0)} attempt(s))"
        )
    return "\n".join(lines)


# ── Recent submissions ─────────────────────────────────────────────────────────

async def fetch_recent_submissions(
    class_id: str, student_id: str, window_days: int
) -> str:
    def _query():
        svc = get_service_client()
        # cutoff date
        from datetime import timedelta
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()
        rows = (
            svc.table("submissions")
            .select(
                "id, attempt_number, status, submitted_at, time_taken_secs, "
                "assignments!inner(title, class_id)"
            )
            .eq("student_id", student_id)
            .eq("assignments.class_id", class_id)
            .gte("submitted_at", cutoff)
            .in_("status", ["graded", "reviewed"])
            .order("submitted_at", desc=True)
            .limit(20)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    lines = ["## Recent Submissions (last 20, graded/reviewed)"]
    for r in rows:
        title = (r.get("assignments") or {}).get("title", "Unknown assignment")
        mins = (
            f"{round(r['time_taken_secs'] / 60, 1)} min"
            if r.get("time_taken_secs")
            else "n/a"
        )
        lines.append(
            f"- {title} | attempt #{r.get('attempt_number', 1)} | "
            f"status: {r.get('status')} | submitted: {(r.get('submitted_at') or '')[:10]} | "
            f"time taken: {mins}"
        )
    return "\n".join(lines)


# ── Question-level results ─────────────────────────────────────────────────────

async def fetch_question_level_results(
    class_id: str, student_id: str, window_days: int
) -> str:
    def _query():
        svc = get_service_client()
        from datetime import timedelta
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()
        rows = (
            svc.table("grading_results")
            .select(
                "marks_awarded, marks_available, trust_layer, amber_flag, "
                "teacher_overridden, teacher_override_marks, ft_applied, "
                "assignment_questions!inner(topic, theme, "
                "  questions!inner(topic, subtopic)), "
                "submissions!inner(student_id, submitted_at, "
                "  assignments!inner(class_id, title))"
            )
            .eq("submissions.student_id", student_id)
            .eq("submissions.assignments.class_id", class_id)
            .gte("submissions.submitted_at", cutoff)
            .limit(50)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    lines = ["## Question-Level Results (last 50 in window)"]
    for r in rows:
        aq = r.get("assignment_questions") or {}
        q = aq.get("questions") or {}
        topic = aq.get("topic") or q.get("topic", "Unknown")
        subtopic = q.get("subtopic", "")
        awarded = (
            float(r["teacher_override_marks"])
            if r.get("teacher_overridden") and r.get("teacher_override_marks") is not None
            else float(r.get("marks_awarded") or 0)
        )
        available = float(r.get("marks_available") or 0)
        flags = []
        if r.get("amber_flag"):
            flags.append("amber")
        if r.get("ft_applied"):
            flags.append("FT")
        if r.get("teacher_overridden"):
            flags.append("overridden")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        label = f"{topic} — {subtopic}" if subtopic else topic
        lines.append(
            f"- {label}: {awarded:.1f}/{available:.1f} | layer: {r.get('trust_layer', 'n/a')}{flag_str}"
        )
    return "\n".join(lines)


# ── Trust-layer breakdown ──────────────────────────────────────────────────────

async def fetch_trust_layer_breakdown(
    class_id: str, student_id: str, window_days: int
) -> str:
    def _query():
        svc = get_service_client()
        from datetime import timedelta
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()
        rows = (
            svc.table("grading_results")
            .select(
                "trust_layer, "
                "submissions!inner(student_id, submitted_at, "
                "  assignments!inner(class_id))"
            )
            .eq("submissions.student_id", student_id)
            .eq("submissions.assignments.class_id", class_id)
            .gte("submissions.submitted_at", cutoff)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    counts: dict[str, int] = {}
    for r in rows:
        layer = r.get("trust_layer") or "unknown"
        counts[layer] = counts.get(layer, 0) + 1

    total = sum(counts.values())
    lines = ["## Trust-Layer Breakdown"]
    for layer, cnt in sorted(counts.items()):
        pct = round(cnt / total * 100, 1) if total else 0
        lines.append(f"- {layer}: {cnt} ({pct}%)")
    lines.append(f"- Total graded questions: {total}")
    return "\n".join(lines)


# ── Attendance & lessons ───────────────────────────────────────────────────────

async def fetch_attendance_and_lessons(class_id: str, student_id: str) -> str:
    def _query():
        svc = get_service_client()
        rows = (
            svc.table("lessons")
            .select("title, start_datetime, status")
            .eq("class_id", class_id)
            .order("start_datetime", desc=True)
            .limit(20)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    lines = ["## Lessons (class schedule, last 20)"]
    for r in rows:
        dt = (r.get("start_datetime") or "")[:10]
        lines.append(f"- {r.get('title', 'Lesson')} | {dt} | {r.get('status', 'unknown')}")
    return "\n".join(lines)


# ── Teacher notes ──────────────────────────────────────────────────────────────

async def fetch_teacher_notes(class_id: str, student_id: str) -> str:
    def _query():
        svc = get_service_client()
        rows = (
            svc.table("lesson_notes")
            .select("note, created_at")
            .eq("student_id", student_id)
            .eq("class_id", class_id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    lines = ["## Teacher Notes (last 20)"]
    for r in rows:
        dt = (r.get("created_at") or "")[:10]
        note = (r.get("note") or "")[:300]
        lines.append(f"- [{dt}] {note}")
    return "\n".join(lines)


# ── Attempt progression ────────────────────────────────────────────────────────

async def fetch_attempt_progression(
    class_id: str, student_id: str, window_days: int
) -> str:
    def _query():
        svc = get_service_client()
        from datetime import timedelta
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()
        rows = (
            svc.table("v_class_attempt_progression")
            .select(
                "assignment_title, attempt_number, submitted_at, "
                "time_taken_secs, marks_awarded, marks_available, score_ratio, "
                "ft_count, override_count"
            )
            .eq("class_id", class_id)
            .eq("student_id", student_id)
            .gte("submitted_at", cutoff)
            .order("submitted_at", desc=True)
            .limit(30)
            .execute()
        )
        return rows.data or []

    rows = await _run(_query)
    if not rows:
        return ""

    lines = ["## Attempt Progression (last 30 in window)"]
    for r in rows:
        ratio = r.get("score_ratio")
        pct = f"{round(float(ratio) * 100, 1)}%" if ratio is not None else "n/a"
        mins = (
            f"{round(r['time_taken_secs'] / 60, 1)} min"
            if r.get("time_taken_secs")
            else "n/a"
        )
        dt = (r.get("submitted_at") or "")[:10]
        lines.append(
            f"- {r.get('assignment_title', 'Assignment')} | attempt #{r.get('attempt_number', 1)} | "
            f"{pct} | time: {mins} | date: {dt} | FT: {r.get('ft_count', 0)} | "
            f"overrides: {r.get('override_count', 0)}"
        )
    return "\n".join(lines)


# ── Access check ───────────────────────────────────────────────────────────────

async def verify_teacher_access(
    class_id: str, student_id: str, teacher_id: str, teacher_role: str
) -> bool:
    """
    Returns True if teacher_id has access to (class_id, student_id).
    Checks:
      - student is enrolled in class
      - teacher owns the class, OR is a co-teacher, OR is a school_admin in the same workspace
    """
    def _query():
        svc = get_service_client()

        # 1. Verify student is enrolled
        enroll = (
            svc.table("class_enrollments")
            .select("id")
            .eq("class_id", class_id)
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        )
        if not enroll.data:
            return False

        if teacher_role == "school_admin":
            # school_admin: verify both teacher and class belong to same workspace
            teacher_ws = (
                svc.table("users")
                .select("workspace_id")
                .eq("id", teacher_id)
                .maybe_single()
                .execute()
            )
            class_teacher = (
                svc.table("classes")
                .select("teacher_id, users!inner(workspace_id)")
                .eq("id", class_id)
                .maybe_single()
                .execute()
            )
            if not teacher_ws.data or not class_teacher.data:
                return False
            t_ws = teacher_ws.data.get("workspace_id")
            c_ws = (class_teacher.data.get("users") or {}).get("workspace_id")
            return t_ws is not None and t_ws == c_ws

        # Direct owner
        owner = (
            svc.table("classes")
            .select("id")
            .eq("id", class_id)
            .eq("teacher_id", teacher_id)
            .maybe_single()
            .execute()
        )
        if owner.data:
            return True

        # Co-teacher
        co = (
            svc.table("class_co_teachers")
            .select("id")
            .eq("class_id", class_id)
            .eq("user_id", teacher_id)
            .not_.is_("accepted_at", "null")
            .maybe_single()
            .execute()
        )
        return co.data is not None

    return await asyncio.to_thread(_query)


# ── Assemble full context ──────────────────────────────────────────────────────

async def assemble_context(
    class_id: str,
    student_id: str,
    options,
) -> tuple[str, list[str]]:
    """
    Fetch all enabled context slices concurrently.
    Returns (assembled_markdown, list_of_context_keys_used).
    """
    tasks: dict[str, object] = {}

    # Identity is always fetched (free, tiny)
    tasks["identity"] = fetch_student_identity(class_id, student_id)

    if options.overall_mastery:
        tasks["overall_mastery"] = fetch_overall_mastery(class_id, student_id)
    if options.topic_mastery:
        tasks["topic_mastery"] = fetch_topic_mastery(class_id, student_id)
    if options.recent_submissions:
        tasks["recent_submissions"] = fetch_recent_submissions(
            class_id, student_id, options.window_days
        )
    if options.question_level_results:
        tasks["question_level_results"] = fetch_question_level_results(
            class_id, student_id, options.window_days
        )
    if options.trust_layer_breakdown:
        tasks["trust_layer_breakdown"] = fetch_trust_layer_breakdown(
            class_id, student_id, options.window_days
        )
    if options.attendance_and_lessons:
        tasks["attendance_and_lessons"] = fetch_attendance_and_lessons(
            class_id, student_id
        )
    if options.teacher_notes:
        tasks["teacher_notes"] = fetch_teacher_notes(class_id, student_id)
    if options.attempt_progression:
        tasks["attempt_progression"] = fetch_attempt_progression(
            class_id, student_id, options.window_days
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    keys = list(tasks.keys())

    blocks: list[str] = []
    used: list[str] = []

    for key, result in zip(keys, results):
        if isinstance(result, Exception):
            logger.warning("[student-context] %s fetch failed: %s", key, result)
            continue
        if result:
            blocks.append(result)
            used.append(key)

    return "\n\n".join(blocks), used
