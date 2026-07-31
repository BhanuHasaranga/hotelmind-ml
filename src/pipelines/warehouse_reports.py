"""Markdown report writers for the warehouse loading milestone.

Separate from src/evaluation/report_writer.py (which is JSON/ML-metrics
shaped): this milestone's deliverables are explicitly Markdown files under
reports/warehouse_loading/.
"""

from pathlib import Path

from src.config.settings import settings


def _write(filename: str, content: str) -> Path:
    path = settings.warehouse_loading_reports_dir_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def write_loading_summary(cleaning_stats: dict, table_row_counts: dict, exec_times: dict) -> Path:
    lines = ["# Loading Summary", ""]

    lines.append("## Cleaning steps")
    lines.append("")
    lines.append("| Step | Rows before | Rows after | Values changed | Reason |")
    lines.append("|---|---|---|---|---|")
    for step_name, s in cleaning_stats.get("steps", {}).items():
        lines.append(
            f"| {step_name} | {s.get('rows_before')} | {s.get('rows_after')} | "
            f"{s.get('values_changed')} | {s.get('reason')} |"
        )

    lines.append("")
    lines.append("## Warehouse table row counts")
    lines.append("")
    lines.append("| Table | Rows |")
    lines.append("|---|---|")
    for table, count in table_row_counts.items():
        lines.append(f"| {table} | {count} |")

    lines.append("")
    lines.append("## Execution times (seconds)")
    lines.append("")
    lines.append("| Stage | Seconds |")
    lines.append("|---|---|")
    for stage, seconds in exec_times.items():
        lines.append(f"| {stage} | {seconds:.2f} |")

    warnings = cleaning_stats.get("warnings", [])
    errors = cleaning_stats.get("errors", [])
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    lines += [f"- {w}" for w in warnings] or ["- None"]
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    lines += [f"- {e}" for e in errors] or ["- None"]
    lines.append("")

    return _write("loading_summary.md", "\n".join(lines))


def write_key_generation(keygen_notes: dict, unmapped_room_codes: set, guest_cluster_stats: dict) -> Path:
    lines = ["# Key Generation Strategy", ""]

    lines.append("All keys are deterministic: identical input always produces an identical key, ")
    lines.append("via `src.pipelines.keygen`. No randomness or UUIDs are used anywhere.")
    lines.append("")

    lines.append("| Key | Algorithm |")
    lines.append("|---|---|")
    for key_name, description in keygen_notes.items():
        lines.append(f"| {key_name} | {description} |")

    lines.append("")
    lines.append("## Unmapped room-type codes encountered")
    lines.append("")
    if unmapped_room_codes:
        lines.append(", ".join(sorted(unmapped_room_codes)))
    else:
        lines.append("None — every observed room-type code mapped to a known tier.")

    lines.append("")
    lines.append("## Guest identity clustering")
    lines.append("")
    lines.append(f"- Distinct guest_keys: {guest_cluster_stats.get('distinct_guest_keys')}")
    lines.append(f"- Total booking rows: {guest_cluster_stats.get('total_rows')}")
    lines.append(
        f"- Average rows per guest_key cluster: {guest_cluster_stats.get('avg_rows_per_cluster', 0):.2f}"
    )
    lines.append("")

    return _write("key_generation.md", "\n".join(lines))


def write_validation_report(validation_results: dict) -> Path:
    lines = ["# Validation Report", ""]

    overall_pass = True

    lines.append("## Row counts / primary key uniqueness")
    lines.append("")
    lines.append("| Table | Row count | PK unique |")
    lines.append("|---|---|---|")
    for table, result in validation_results.get("tables", {}).items():
        pk_unique = result.get("pk_unique")
        overall_pass = overall_pass and bool(pk_unique)
        lines.append(f"| {table} | {result.get('row_count')} | {pk_unique} |")

    lines.append("")
    lines.append("## NULL violations")
    lines.append("")
    lines.append("| Table | Column | Null count |")
    lines.append("|---|---|---|")
    any_null_violation = False
    for table, result in validation_results.get("tables", {}).items():
        for col, count in result.get("null_violations", {}).items():
            if count:
                any_null_violation = True
                overall_pass = False
            lines.append(f"| {table} | {col} | {count} |")

    lines.append("")
    lines.append("## Foreign key integrity")
    lines.append("")
    lines.append("| FK | Orphan count | Sample orphans |")
    lines.append("|---|---|---|")
    for fk_name, result in validation_results.get("fk_checks", {}).items():
        orphan_count = result.get("orphan_count", 0)
        overall_pass = overall_pass and orphan_count == 0
        lines.append(f"| {fk_name} | {orphan_count} | {result.get('sample_orphans')} |")

    lines.append("")
    lines.append(f"## Overall result: {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")

    return _write("validation_report.md", "\n".join(lines))


def write_mapping_summary(assumptions: dict) -> Path:
    lines = ["# Mapping & Assumption Summary", ""]
    for title, body in assumptions.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")

    return _write("mapping_summary.md", "\n".join(lines))


def _md_cell(text: str) -> str:
    """Flattens a value to a single-line, pipe-safe markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def write_lineage_report(raw_count: int, cleaned_count: int, warehouse_counts: dict, db_counts: dict | None) -> Path:
    lines = ["# Data Lineage Report", ""]
    lines.append("| Stage | Table/File | Row Count | Notes |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Raw | hotel_bookings.csv | {raw_count} | - |")
    lines.append(
        f"| Cleaned | hotel_bookings_clean.parquet | {cleaned_count} | "
        "after dedupe + negative-adr drop |"
    )
    for table, count in warehouse_counts.items():
        lines.append(f"| Warehouse Files | {table} | {count} | local parquet, always written |")

    if db_counts is None:
        lines.append("| Database | (all tables) | — | not loaded to DB (--write-db not passed) |")
    else:
        for table, result in db_counts.items():
            count = result.get("rows_written", 0)
            note = _md_cell(result.get("note", "-"))
            lines.append(f"| Database | {table} | {count} | {note} |")

    lines.append("")

    return _write("lineage_report.md", "\n".join(lines))
