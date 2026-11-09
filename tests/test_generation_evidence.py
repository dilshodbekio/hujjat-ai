from dataclasses import replace

import pytest

from app.generation.context import build_context, validate_and_render
from app.generation.evidence import extract_evidence_span
from app.generation.models import AnswerValidationError, ModelAnswer
from app.retrieval.vector_store import SearchResult


@pytest.mark.parametrize("source,quote", [
    ("Sick leave\n5 paid days for the whole internship.", "Sick leave: 5 paid days for the whole internship."),
    ("  Sick leave\r\n5 paid days for the whole internship.", "Sick leave: 5 paid days for the whole internship."),
    ("A\nB", "A B"),
    ("Sick leave\n5 paid days.\nOther section", "5 paid days."),
    ("Sick leave\n5 paid days.", "Sick leave\n5 paid days."),
])
def test_source_span_round_trip(source, quote):
    span = extract_evidence_span(source, quote)
    assert span is not None
    assert span.text == source[span.start_char:span.end_char]
    assert ":" not in span.text


@pytest.mark.parametrize("source,quote", [
    ("Sick leave 5 paid days.", "Sick leave: 5 paid days."),  # no heading newline
    ("Sick leave\n\n5 paid days.", "Sick leave: 5 paid days."),  # paragraph break
    ("sick leave\n5 paid days.", "sick leave: 5 paid days."),  # not label form
    ("Not eligible\n5 paid days.", "Not eligible: 5 paid days."),
    ("Sick leave\n5 paid days.", "Sick leave: 6 paid days."),
    ("Sick leave\n5 paid days.", "Sick leave: 15 paid days."),
    ("Sick leave\n5 unpaid days.", "Sick leave: 5 paid days."),
    ("Sick leave\nNo paid days.", "Sick leave: paid days."),
    ("Employees are not eligible.", "Employees are eligible."),
    ("Employees are not eligible.", "eligible"),
    ("Employees are unpaid.", "paid"),
    ("Deadline: 15 December 2026.", "Deadline: 15 December 2027."),
    ("Deadline: 15 December 2026.", "Deadline 15 December 2026."),
    ("John Smith is eligible.", "Jane Smith is eligible."),
    ("John Smith is eligible.", "john Smith is eligible."),
    ("Anna-Marie is eligible.", "Marie"),
    ("Employees receive 15 days.", "5 days"),
    ("Employees receive -5 days.", "5 days"),
    ("Employees receive - 5 days.", "5 days"),
    ("Employees receive 1.5 days.", "5 days"),
    ("Employees receive 1.5 days.", "1"),
    ("Employees receive 10,000 units.", "10"),
    ("The rate is 1/2.", "1"),
    ("The fee is £5.", "5"),
    ("Sick leave\n5 paid days.", "Sick leave; 5 paid days."),
    ("No entitlement\n5 days.", "Entitlement: 5 days."),
    ("Sick leave\n5 days.", "..."),
])
def test_manipulated_evidence_rejected(source, quote):
    assert extract_evidence_span(source, quote) is None


def test_colon_match_cannot_bypass_source_ids_or_citation_protections(sample_chunks):
    chunk = replace(sample_chunks[0], text="Sick leave\n5 paid days for the whole internship.")
    context = build_context([SearchResult(chunk, 1.)])
    for source_id, text in [("S99", "Interns receive 5 paid sick days."),
                            ("S1", "Interns receive 5 paid sick days. [fake.pdf, page 999]")]:
        answer = ModelAnswer.model_validate({"status": "answered", "claims": [{
            "text": text, "evidence": [{"source_id": source_id,
                "quote": "Sick leave: 5 paid days for the whole internship."}]}]})
        with pytest.raises(AnswerValidationError):
            validate_and_render(answer, context)


def test_one_valid_formatting_match_does_not_hide_manipulated_claim(sample_chunks):
    chunk = replace(sample_chunks[0], text="Sick leave\n5 paid days for the whole internship.")
    context = build_context([SearchResult(chunk, 1.)])
    answer = ModelAnswer.model_validate({"status": "answered", "claims": [
        {"text": "Interns receive 5 days.", "evidence": [{"source_id": "S1",
            "quote": "Sick leave: 5 paid days for the whole internship."}]},
        {"text": "Interns receive 15 days.", "evidence": [{"source_id": "S1",
            "quote": "Sick leave: 15 paid days for the whole internship."}]}]})
    with pytest.raises(AnswerValidationError):
        validate_and_render(answer, context)


def test_exact_live_failure_and_heading_normalization_regression():
    # Exact live failure quote from northwind_handbook_v3.2.pdf Appendix C (diagnostic 20261109T135144Z)
    source = (
        "Appendix C – Interns and trainees\n"
        "Annual leave\n"
        "1 working day for each full month of the internship.\n"
        "Sick leave\n"
        "5 paid days for the whole internship.\n"
        "Work pattern\n"
        "Office-based, with up to 2 remote days per week if the mentor agrees.\n"
    )
    quote = "Sick leave: 5 paid days for the whole internship."
    span = extract_evidence_span(source, quote)
    assert span is not None
    assert span.text == "Sick leave\n5 paid days for the whole internship."
    assert span.text == source[span.start_char:span.end_char]
    assert ":" not in span.text

    # Numbered heading section repair
    numbered_source = "3.2 Sick leave\n10 paid sick days per calendar year for all employees."
    numbered_quote = "3.2 Sick leave: 10 paid sick days per calendar year for all employees."
    numbered_span = extract_evidence_span(numbered_source, numbered_quote)
    assert numbered_span is not None
    assert numbered_span.text == numbered_source
    assert ":" not in numbered_span.text

    # Punctuation/dash in heading repair
    dash_source = "Appendix C – Interns and trainees\nDuration\nUp to 6 months; set in the internship agreement."
    dash_quote = "Appendix C – Interns and trainees: Duration"
    dash_span = extract_evidence_span(dash_source, dash_quote)
    assert dash_span is not None
    assert dash_span.text == "Appendix C – Interns and trainees\nDuration"

    # Harmless typography differences: curly apostrophe in source -> straight apostrophe in quote
    curly_source = "Where the law and the employee’s employing entity conflict, the law prevails."
    straight_quote = "employee's employing entity conflict"
    curly_span = extract_evidence_span(curly_source, straight_quote)
    assert curly_span is not None
    assert curly_span.text == "employee’s employing entity conflict"  # Preserves exact source slice with curly char

    # Harmless typography differences: curly quotes in source -> straight quotes in quote
    quote_source = 'roles tagged “Priority Hire” on the careers page'
    ascii_quote = 'roles tagged "Priority Hire" on the careers page'
    quote_span = extract_evidence_span(quote_source, ascii_quote)
    assert quote_span is not None
    assert quote_span.text == 'roles tagged “Priority Hire” on the careers page'

    # Harmless outer quotation marks in model quote
    wrapped_quote = '"Sick leave: 5 paid days for the whole internship."'
    wrapped_span = extract_evidence_span(source, wrapped_quote)
    assert wrapped_span is not None
    assert wrapped_span.text == "Sick leave\n5 paid days for the whole internship."


@pytest.mark.parametrize("source,quote", [
    # Adversarial omitted negations:
    ("Employees are not eligible for any bonus.", "eligible for any bonus."),
    ("Contractors receive no paid leave under this agreement.", "paid leave under this agreement."),
    ("Travel is not permitted without prior manager approval.", "permitted without prior manager approval."),
    ("Employees must never disclose confidential information.", "disclose confidential information."),
    ("Neither interns nor contractors are eligible for stock options.", "interns nor contractors are eligible for stock options."),
    ("Leave without pay is strictly monitored.", "pay is strictly monitored."),
    # Adversarial headings with negations must NOT receive synthetic colons:
    ("Not eligible\n5 paid days for the whole internship.", "Not eligible: 5 paid days for the whole internship."),
    ("No entitlement\nFull salary payout on departure.", "No entitlement: Full salary payout on departure."),
    ("Without exception\nEmployees must register attendance.", "Without exception: Employees must register attendance."),
    # Colon abuse: attempting to quote only the synthetic colon or start/end on it:
    ("Sick leave\n5 paid days for the whole internship.", ": 5 paid days for the whole internship."),
    ("Sick leave\n5 paid days for the whole internship.", "Sick leave:"),
    # Adversarial altered facts (numbers, dates, names, partial tokens):
    ("Employees receive 10 paid sick days.", "Employees receive 12 paid sick days."),
    ("Effective 1 June 2026.", "Effective 1 July 2026."),
    ("Bonus of $750 for every eligible hire.", "750 for every eligible hire."),
    ("Applicants with Anna-Marie as referee.", "Marie as referee."),
])
def test_adversarial_evidence_manipulations_strictly_rejected(source, quote):
    assert extract_evidence_span(source, quote) is None
