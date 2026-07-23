from datetime import UTC, datetime

import pytest

from pymixef.pharmacometrics import (
    DoseAmountStatus,
    EventTable,
    EventType,
    EventValidationError,
    canonicalize_events,
)


def test_canonicalization_is_immutable_sorted_and_audited() -> None:
    source = [
        {"ID": "A", "TIME": 1, "EVID": 0, "DV": 4.2},
        {"ID": "A", "TIME": 0, "EVID": 1, "AMT": 100, "CMT": 1},
    ]
    table = canonicalize_events(source)
    assert isinstance(table.events, tuple)
    assert [event.TIME for event in table] == [0.0, 1.0]
    assert [event.row_id for event in table] == ["row-00000001", "row-00000000"]
    assert table[1].MDV == 0
    assert table[0].MDV == 1
    assert table.source_count == 2
    assert table.to_source_records() == source
    assert all(entry.code == "EVENT-CANONICALIZED-001" for entry in table.audit)
    with pytest.raises(TypeError):
        table[0].covariates["weight"] = 70  # type: ignore[index]
    # Input is never rewritten.
    assert "ROW_ID" not in source[0]


def test_same_time_order_is_explicit_and_stable() -> None:
    table = canonicalize_events(
        [
            {"ID": 1, "TIME": 0, "EVID": 0, "DV": 10},
            {"ID": 1, "TIME": 0, "EVID": "dose", "AMT": 10},
            {"ID": 1, "TIME": 0, "EVID": "covariate", "COVARIATES": {"WT": 80}},
            {"ID": 1, "TIME": 0, "EVID": "reset"},
            {"ID": 1, "TIME": 0, "EVID": "infusion_stop", "RATE": 2},
        ]
    )
    assert [event.evid for event in table] == [
        EventType.RESET,
        EventType.COVARIATE,
        EventType.INFUSION_STOP,
        EventType.DOSE,
        EventType.OBSERVATION,
    ]


def test_addl_expansion_has_stable_ids_and_provenance() -> None:
    table = canonicalize_events(
        [
            {
                "ID": "S1",
                "TIME": 0,
                "EVID": 1,
                "AMT": 50,
                "ADDL": 2,
                "II": 12,
                "ROW_ID": "dose-a",
            }
        ]
    )
    expanded = table.expand_additional()
    assert [event.TIME for event in expanded] == [0.0, 12.0, 24.0]
    assert [event.row_id for event in expanded] == [
        "dose-a",
        "dose-a:addl:1",
        "dose-a:addl:2",
    ]
    assert all(event.ADDL == 0 for event in expanded)
    assert all(event.source_row_id == "dose-a" for event in expanded)
    assert sum(entry.code == "EVENT-ADDL-EXPANDED-001" for entry in expanded.audit) == 2
    # Expansion is idempotent.
    assert expanded.expand_additional().to_records() == expanded.to_records()


def test_duration_derived_infusion_and_explicit_stop() -> None:
    table = canonicalize_events([{"ID": 1, "TIME": 2, "EVID": 1, "AMT": 120, "DUR": 3, "CMT": 1}])
    start = table[0]
    assert pytest.approx(40) == start.RATE
    assert start.is_infusion
    expanded = table.expand_infusions()
    assert [event.kind for event in expanded] == ["infusion_start", "infusion_stop"]
    assert pytest.approx(5) == expanded[1].TIME
    assert pytest.approx(40) == expanded[1].RATE
    assert expanded[1].generated

    explicitly_stopped = canonicalize_events(
        [
            {"ID": 1, "TIME": 0, "EVID": 1, "AMT": 20, "RATE": 10},
            {"ID": 1, "TIME": 2, "EVID": "infusion_stop", "RATE": 10},
        ]
    ).expand_infusions()
    assert sum(event.evid == EventType.INFUSION_STOP for event in explicitly_stopped) == 1


def test_inconsistent_infusion_is_rejected() -> None:
    with pytest.raises(EventValidationError, match="inconsistent"):
        canonicalize_events(
            [
                {
                    "ID": 1,
                    "TIME": 0,
                    "EVID": 1,
                    "AMT": 100,
                    "RATE": 10,
                    "DUR": 5,
                }
            ]
        )


def test_missing_and_zero_dose_are_distinct() -> None:
    with pytest.raises(EventValidationError, match="AMT_STATUS"):
        canonicalize_events([{"ID": 1, "TIME": 0, "EVID": 1}])
    unknown = canonicalize_events([{"ID": 1, "TIME": 0, "EVID": 1, "AMT_STATUS": "unknown"}])
    assert unknown[0].amount is None
    assert unknown[0].amount_status == DoseAmountStatus.UNKNOWN
    table = canonicalize_events([{"ID": 1, "TIME": 0, "EVID": 1, "AMT": 0}])
    assert table[0].AMT == 0.0
    assert table[0].amount_status == DoseAmountStatus.RECORDED

    observation = canonicalize_events([{"ID": 1, "TIME": 1, "EVID": 0}])
    assert observation[0].amount_status == DoseAmountStatus.NOT_APPLICABLE


def test_additional_doses_require_positive_interval() -> None:
    with pytest.raises(EventValidationError, match="II"):
        canonicalize_events([{"ID": 1, "TIME": 0, "EVID": 1, "AMT": 10, "ADDL": 1}])


def test_timezone_aware_calendar_time_conversion_is_audited() -> None:
    point = datetime(2026, 7, 22, 12, tzinfo=UTC)
    table = canonicalize_events([{"ID": 1, "TIME": point, "EVID": 0}])
    assert point.timestamp() == table[0].TIME
    assert any(entry.code == "EVENT-TIME-CONVERTED-001" for entry in table.audit)
    with pytest.raises(EventValidationError, match="timezone-aware"):
        canonicalize_events([{"ID": 1, "TIME": datetime(2026, 7, 22, 12), "EVID": 0}])


def test_multiple_subject_selection_and_structured_array_round_trip() -> None:
    table = EventTable.from_records(
        [
            {"ID": "B", "TIME": 0, "EVID": 0},
            {"ID": "A", "TIME": 0, "EVID": 0},
        ]
    )
    assert table.subjects == ("A", "B")
    assert len(table.for_subject("A")) == 1
    records = table.to_records()
    assert EventTable.from_records(records).subjects == ("A", "B")
