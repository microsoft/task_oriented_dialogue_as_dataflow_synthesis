from dataflow.core.lispress import (
    _try_round_trip,
    lispress_to_program,
    parse_lispress,
    program_to_lispress,
    render_pretty,
)
from dataflow.core.program import Program, TypeName
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
        :subject (?= "makeup artist")))))""",
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
          (Constraint[Event] :subject (?= "sales conference"))
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
                    (Constraint[Event] :subject (?~= "lunch"))
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
                (PeriodDuration :duration (toHours 4))))
            :subject (?= "dinner at foo")))))))""",
    # tests that whitespace is preserved inside a quoted string,
    # as opposed to tokenized and then joined with a single space.
    '"multi\\tword  quoted\\nstring"',
    '"i got quotes\\""',
    '#(PersonName "multi\\tword  quoted\\nstring")',
    '#(PersonName "i got quotes\\"")',
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
      :number 1)))""",
    # META_CHAR expression
    """
(Yield
  (^(Long) >
    ^Long
    (size
      (QueryEventResponse.results
        (FindEventWrapperWithDefaults
          (EventDuringRange
            (^(Event) EmptyStructConstraint)
            (ThisWeekend)))))
    0L))""",
    # Long VALUE_CHAR expression
    """
(Yield
  (==
    #(PersonName
      "veeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeerylong")
    #(PersonName "short")))
""",
    # nested lambdas
    """
(action
  (Inform
    (^(Navigation) Find
      :focus (Some
        (Constraint.apply
          (lambda
            (^Navigation x0)
            (allows
              (Constraint.apply
                (lambda
                  (^AppleDuration x1)
                  (allows (?= 10) (AppleDuration.minutes x1))))
              (Navigation.travelTime x0))))))))
""",
    """
(action
  (Prompt
    (^(Message) Create
      :object (Some
        (Constraint.apply
          (lambda
            (^Message x0)
            (allows
              (Constraint.apply
                (lambda
                  (^Contact x1)
                  (allows
                    (Constraint.apply
                      (lambda
                        (^Person x2)
                        (allows (^(String) always) (Person.nameHint x2))))
                    (Contact.person x1))))
              (Message.recipients x0))))))))
""",
    # nested lambda type-arg
    """
(plan
  (revise
    (^(Unit) Path.apply "Create")
    (^((Constraint Person)) Path.apply
      "object.recipients.person")
    (lambda
      (^(Constraint Person) x0)
      (& x0 (Person.nameHint_? (?= "Payne"))))))
""",
    # lambda arg that is never referenced
    """
(plan
  (revise
    (^(Unit) Path.apply "Create")
    (^((Option (Constraint ConfirmStatus))) Path.apply
      "confirmation")
    (lambda
      (^(Option (Constraint ConfirmStatus)) x0)
      (Some (?= (ConfirmStatus.Accepted))))))
""",
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


def round_trip_through_program(s):
    sexp = parse_lispress(s)
    program, _ = lispress_to_program(sexp, 0)
    round_tripped_sexp = program_to_lispress(program)
    return render_pretty(round_tripped_sexp, max_width=60)


def test_surface_to_program_round_trips():
    """
    Goes all the way to `Program` and so is stricter
    than `test_surface_to_sexp_round_trips`.
    """
    for surface_string in surface_strings:
        s = surface_string.strip()
        round_tripped_surface_string = round_trip_through_program(s)
        assert round_tripped_surface_string == s


def test_program_to_lispress_with_quotes_inside_string():
    # a string with a double-quote in it
    v, _ = mk_value_op(value='i got quotes"', schema="String", idx=0)
    program = Program(expressions=[v])
    rendered_lispress = render_pretty(program_to_lispress(program))
    assert rendered_lispress == '"i got quotes\\""'
    sexp = parse_lispress(rendered_lispress)
    round_tripped, _ = lispress_to_program(sexp, 0)
    assert round_tripped == program


def test_bare_values():
    assert _try_round_trip("0L") == "0L"
    assert _try_round_trip("0") == "0.0"
    assert _try_round_trip("0.0") == "0.0"
    assert _try_round_trip("#(Number 0)") == "0.0"
    assert _try_round_trip("#(Number 0.0)") == "0.0"


def test_typenames():
    roundtrip = _try_round_trip("^Number (^(String) foo (bar) ^Bar (bar))")
    assert roundtrip == "^Number (^(String) foo (bar) ^Bar (bar))"


def test_typename_with_args():
    roundtrip = _try_round_trip("^(Number Foo) (^(String) foo (bar) ^Bar (bar))")
    assert roundtrip == "^(Number Foo) (^(String) foo (bar) ^Bar (bar))"


def test_sorts_named_args():
    # TODO: scary: named
    roundtrip = _try_round_trip("(Foo :foo 1.0 :bar 3.0)")
    assert roundtrip == "(Foo :bar 3.0 :foo 1.0)"


def test_mixed_named_and_positional_args():
    # TODO: scary: named
    roundtrip = _try_round_trip("(Foo 1.0 2.0 :bar 3)")
    assert roundtrip == "(Foo 1.0 2.0 :bar 3.0)"


def test_number_float():
    lispress = "(Yield (> (a) 0.0))"
    assert _try_round_trip(lispress) == lispress
    assert _try_round_trip("(Yield (> (a) 0))") == lispress
    assert _try_round_trip("(toHours 4)") == "(toHours 4.0)"


def test_bool():
    assert _try_round_trip("(toHours true)") == "(toHours true)"


def test_string():
    assert _try_round_trip('(+ (a) #(String "b"))') == '(+ (a) "b")'
    assert _try_round_trip('(+ (a) #(PersonName "b"))') == '(+ (a) #(PersonName "b"))'


def test_escaped_name():
    string = "(a\\ b)"
    assert parse_lispress(string) == ["a b"]
    assert _try_round_trip(string) == string


def test_strip_copy_strings():
    assert _try_round_trip('#(String " Tom ")') == '"Tom"'
    assert _try_round_trip('" Tom "') == '"Tom"'


def test_type_args_in_program():
    lispress_str = "(^(PleasantryCalendar) EmptyStructConstraint)"
    lispress = parse_lispress(lispress_str)
    program, _ = lispress_to_program(lispress, 0)
    assert len(program.expressions) == 1
    assert program.expressions[0].type_args == [TypeName("PleasantryCalendar", ())]
    assert program.expressions[0].type is None


def test_fully_typed_reference():
    # This is a regression test, formerly an Exception was thrown.
    s = "(lambda (^Unit x0) ^Unit x0)"
    # The second type ascription is not retained because x0 is represented by
    # a single node in a Program, and so can't have two separate type
    # ascriptions.
    assert round_trip_through_program(s) == "(lambda (^Unit x0) x0)"
