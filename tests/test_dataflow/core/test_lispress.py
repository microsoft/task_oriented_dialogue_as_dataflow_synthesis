from dataflow.core.lispress import (
    lispress_to_program,
    parse_lispress,
    program_to_lispress,
    render_pretty,
    try_round_trip,
)
from dataflow.core.program import Program
from dataflow.core.program_utils import mk_value_op

surface_strings = [
    """
(Yield
  :output (createCommitEventWrapper
    (createPreflightEventWrapper
      (Constraint[Event]
        :start (?=
          (dateAtTimeWithDefaults
            (nextDOW #(DayOfWeek "THURSDAY"))
            (numberPM #(Int 5))))
        :subject (?= #(String "makeup artist"))))))""",
    # Contains a sugared `get` (`(:start ...)`)
    """
(Yield
  :output (:start
    (findNextEvent
      (Constraint[Event]
        :attendees (attendeeListHasRecipientConstraint
          (recipientWithNameLike
            (Constraint[Recipient])
            #(PersonName "Elaine")))))))""",
    """
(do
  (Yield
    :output (Execute :intension (confirmAndReturnAction)))
  (Yield
    :output (createCommitEventWrapper
      (createPreflightEventWrapper
        (eventAllDayOnDate
          (Constraint[Event]
            :subject (?= #(String "sales conference")))
          (nextDayOfMonth (today) #(Int 29)))))))""",
    """
(Yield
  :output (WeatherQueryApi
    :place (atPlace (here))
    :time (Constraint[DateTime]
      :date (?= (nextDOW #(DayOfWeek "SATURDAY"))))))""",
    """(fenceGibberish)""",
    """
(let
  (x0
    (Execute
      :intension (chooseCreateEvent
        #(Int 1)
        (refer (actionIntensionConstraint))))
    x1
    (:end (:item x0)))
  (do
    (Yield :output x0)
    (Yield
      :output (deleteCommitEventWrapper
        (deletePreflightEventWrapper
          (:id
            (singleton
              (:results
                (findEventWrapperWithDefaults
                  (eventOnDateAfterTime
                    (Constraint[Event] :subject (?~= #(String "lunch")))
                    (:date x1)
                    (:time x1)))))))))))""",
    # Includes a `get` that should not be desugared,
    # because the path ("item two") is not a valid identifier (contains whitespace):
    """
(let
  (x0 (Execute :intension (confirmAndReturnAction)))
  (do
    (Confirm :value (GetSalientUnconfirmedValue))
    (Yield :output x0)
    (Yield
      :output (CreateCommitEventWrapper
        :event (CreatePreflightEventWrapper
          :constraint (Constraint[Event]
            :start (?=
              (adjustByPeriodDuration
                (:end (get x0 #(Path "item two")))
                (PeriodDuration :duration (toHours #(Number 4)))))
            :subject (?= #(String "dinner at foo"))))))))""",
    # tests that whitespace is preserved inside a quoted string,
    # as opposed to tokenized and then joined with a single space.
    '(#(String "multi\\tword  quoted\\nstring"))',
    '(#(String "i got quotes\\""))',
    # tests that empty plans are handled correctly
    "()",
    # regression test that no whitespace is inserted between "#" and "(".
    # `#(PersonName "Tom")` was being rendered with whitespace.
    """
(Yield
  :output (:start
    (FindNumNextEvent
      :constraint (Constraint[Event]
        :attendees (AttendeeListHasRecipient
          :recipient (FindManager
            :recipient (Execute
              :intension (refer
                (extensionConstraint
                  (RecipientWithNameLike
                    :constraint (Constraint[Recipient])
                    :name #(PersonName "Tom"))))))))
      :number #(Number 1))))""",
]


def test_surface_to_sexp_round_trips():
    """
    Tests that parsing a Lispress surface string into an S-expression
    and then pretty printing it is a no-op.
    """
    for surface_string in surface_strings:
        surface_string = surface_string.strip()
        lispress = parse_lispress(surface_string)
        round_tripped_surface_string = render_pretty(lispress)
        assert round_tripped_surface_string == surface_string


def test_surface_to_program_round_trips():
    """
    Goes all the way to `Program` and so is stricter
    than `test_surface_to_sexp_round_trips`.
    """
    for surface_string in surface_strings:
        surface_string = surface_string.strip()
        sexp = parse_lispress(surface_string)
        program, _ = lispress_to_program(sexp, 0)
        round_tripped_sexp = program_to_lispress(program)
        assert round_tripped_sexp == sexp
        round_tripped_surface_string = render_pretty(round_tripped_sexp, max_width=60)
        assert round_tripped_surface_string == surface_string


def test_program_to_lispress_with_quotes_inside_string():
    # a string with a double-quote in it
    v, _ = mk_value_op(value='i got quotes"', schema="String", idx=0)
    program = Program(expressions=[v])
    rendered_lispress = render_pretty(program_to_lispress(program))
    assert rendered_lispress == '(#(String "i got quotes\\""))'
    round_tripped, _ = lispress_to_program(parse_lispress(rendered_lispress), 0)
    assert round_tripped == program


def test_bare_values():
    surface_string = "0"
    round_tripped = try_round_trip(surface_string)
    assert round_tripped == "(#(Number 0))"
