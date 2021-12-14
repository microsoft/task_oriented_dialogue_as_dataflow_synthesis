# Program Semantics

SMCalFlow programs are written in the [Lispress](./README-LISPRESS.md)
language. Executing those programs requires a *library* that defines
the execution logic for each function. Although a library is not required for
the semantic parsing task, knowing the program
semantics can often help us understanding the dataset better, and
sometimes helps making better modeling decisions.

In this document, we use some representative programs to explain how
calendar events are created, queried, updated and deleted in SMCalFlow.
We also illustrate how context reference and revision works by a
concrete example.

## Create Event

Here is a program for the utterance "Create a work meeting on
Monday at 11 AM":
```clojure
(Yield 
  (CreateCommitEventWrapper 
    (CreatePreflightEventWrapper 
      (& 
        (Event.subject_? 
          (?= "work meeting")) 
        (Event.start_? 
          (& 
            (DateTime.date_? (?= (NextDOW (Monday)))) 
            (DateTime.time_? (?= (NumberAM 11L)))))))))
```

This program constructs a conjunction of constraints of type `(Constraint Event)`
containing two sub-constraints, one on each of the `start` and `subject` fields. 
Each sub-constraint can be compositional too. For example, the `start` field 
(which represents constraint on the start time of the event) has two sub-constraints: 
`date` and `time`. In general, a method called `Foo.bar_?` takes a single argument of
type `(Constraint Bar)` where `Bar` is the type of the field `bar` on `Foo` and
returns a `(Constraint Foo)`.

Let's take a closer look at the `date` sub-field. This field contains
a `Date` constraint constructed by the `?=` function.
Here, `?=` means "exact match". It takes a
reference object of type `T` and returns a constraint
that is satisfied if the input value is equal to the
reference. In this example, the `date` constraint is satisfied if
the event date is equal to "next Monday". Similarly, the `time` constraint
is satisfied if the event's start time is equal to 11 AM, and the `subject`
constraint is satisfied if the event's subject is equal to "work meeting".

Why do we want to use *constraints*, instead of concrete values, to
represent date, time and subject?
It is because that constraints are flexible enough to represent
the ambiguity in natural language. For example, if the user says "Create a work meeting
on Monday morning" or "Create a work meeting after the lunch",
she doesn't specify any concrete time, but both "morning" and
"after the lunch" can be represented as time constraints.

After the `Event` constraint is constructed, it is sent to
`CreatePreflightEventWrapper`, which solves a constraint satisfaction problem to figure
out the actual `Event` that the user wants to create. If a concrete solution
cannot be determined, then the solver will either inject some
heuristic preference (e.g. the meeting is preferred to be scheduled in
the business hours and the default length is 30 minutes) to tighten the
constraint, or throw an `DisambiguationError` which triggers a
follow-up dialogue turn for disambiguation (See Section 5 of
[the paper](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00333)).

If a concrete `Event` can be resolved, then it is sent to `CreateCommitEventWrapper`
to be added to the database. This step incurs side effects to the
outside world, namely changing the user's calendar. Before writing,
the function will search the dataflow graph to see if the event creation
has been "confirmed" by the user. In this example, it is not confirmed
yet, therefore an `UnconfirmedError` will be thrown, which triggers an
agent utterance like: "Does this look right? \[showing an event card\]".

The functions `CreatePreflightEventWrapper` and `CreateCommitEventWrapper`
represent a common two-step process to modify a user's calendar.
In a pre-processing step, the "Preflight" function takes a user constraint
and runs constraint inference to figure out the actual change that the user
wants to make, then the "Commit" function commits that change to the
database. The "Wrapper" suffix suggests that both functions wrap a series of
sub-steps in them, as we described above. The same naming convention
is used in the update and the deletion programs too.

If the user says "Yes" in the next turn, then the confirmation program
will be like:
```clojure
(Yield (Execute (^(Dynamic) ConfirmAndReturnAction)))
```
The `ConfirmAndReturnAction` function puts a "confirm" token to the
dataflow graph and returns the last turn's program. The `Execute`
function then re-executes that program. This time, since the creation
is confirmed, the `CreateCommitEventWrapper` call will succeed and
the event will be added to the user's calendar. `Dynamic` is a special type
(inspired by [Haskell](https://hackage.haskell.org/package/base-4.15.0.0/docs/Data-Dynamic.html))
that we use whenever a type variable is unconstrained.

## Query Event

The user can query events in the calendar. If the user says
"Where is the avocado festival?", then the program is:
```clojure
(Yield 
  (Event.location 
    (singleton 
      (QueryEventResponse.results 
        (FindEventWrapperWithDefaults (Event.subject_? (?~= "avocado festival")))))))
```
This program builds an `Event` constraint requiring the event's
subject to fuzzily match the string "avocado festival" (the `?~=`
function returns a "fuzzy match" constraint). It is then
sent to the `FindEventWrapperWithDefaults` for querying the database.
The "WithDefault" suffix suggests that the function may inject some
default settings if they are not specified by the constraint,
such as the default time zone and the default ordering of search results.

The query will return a response object, whose `results` field contains
a list of events that matches the constraint. The program
calls the `singleton` function to convert the list to an element, and
in the last step, describes its `location` property. If no event matches
the constraint, or if there are multiple matches, then the `singleton`
function will throw a `NonSingletonListError` which triggers the error
handling mechanism (See Section 5 of
[the paper](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00333)).

## Update Event

The user can update existing events. For example, if
the user says "Change my meeting tomorrow to 4 PM", then the program is:
```clojure
(Yield 
  (UpdateCommitEventWrapper 
    (UpdatePreflightEventWrapper 
      (Event.id 
        (singleton 
          (QueryEventResponse.results 
            (FindEventWrapperWithDefaults
              (Event.start_? 
                (DateTime.date_? (?= (Tomorrow)))) 
              )
            )
          )
        )
        (Event.start_? 
          (DateTime.time_? 
            (?= (NumberPM 4L))
          )
        )
      )
    )
  )
)
```
In this program, `UpdatePreflightEventWrapper` takes two
arguments: `id` and `update`. The `id` argument specifies the event id
to be updated. It is obtained through querying the database
via a constraint, which requires the event's date to be tomorrow.
The `update` argument encodes the requirements for the update. Here,
it requires the new event's start time to be at 4 PM. The
`UpdatePreflightEventWrapper` function uses the `id` to extract an
event from the database, then it uses the `update` constraint to revise it.
The revision guarantees that the new event will satisfy the user constraint, but
preserving the original properties as long as they don't
conflict with the user constraint. For example, an "1-hour
brainstorm meeting at 3 PM" will be revised to be an "1-hour
brainstorm meeting at 4 PM".

If the revision is successful, then the new event will be sent to
`UpdateCommitEventWrapper` to be written to the database.
Similar to creation, this function will require a user confirmation
before it actually takes the action.

## Delete Event

Deletion is similar to update, except that you don't
need to specify a constraint for the new event. If the user says
"Delete my meeting with Emma", then the program is:
```clojure
(Yield 
  (DeleteCommitEventWrapper 
    (DeletePreflightEventWrapper 
      (Event.id 
        (singleton 
          (QueryEventResponse.results 
            (FindEventWrapperWithDefaults 
              (Event.attendees_?
                (AttendeeListHasRecipientConstraint 
                  (RecipientWithNameLike (PersonName.apply "Emma"))
                )
              )
            )
          )
        )
      )
    )
  )
)
```
The program builds an `Event` consraint specifying that the event's attendee
list must contain "Emma". The constraint is sent to `FindEventWrapperWithDefaults`
to query the database. If the event can
be found, then its id is sent to `DeletePreflightEventWrapper` (which
exists only to make the program's structure similar to update)
and `DeleteCommitEventWrapper` (which performs the deletion). A user
confirmation is required.

## Refer and Revise

Sometimes the user may want to revise a previous request. If
the user asks "Anything earlier?" in a follow-up turn, then we interpret
it as "redo an existing computation but constrain the event time to
be earlier than a previously mentioned event's time". The program is:
```clojure
(Yield 
  (Execute 
    (ReviseConstraint 
      (refer (^(Dynamic) roleConstraint (Path.apply "output")))
      (^(Event) ConstraintTypeIntension)
      (Event.start_? 
        (DateTime.time_? 
          (?< 
            (Execute
              (refer 
                (& 
                  (roleConstraint (Path.apply "start"))
                  (extensionConstraint (^(Time) EmptyStructConstraint))
                )
              )
            )
          )
        )
      )
    )
  )
)
```
This program calls `refer` to extract a salient `Time` that was
the start time of a previous event. Here, the `roleConstraint`
requires that the referred computation is in the `start` field of
a structure, and the `extensionConstraint` constrains the result of
that computation. In this example, it only requires the value type
to be `Time`.

The program then constructs a new `Event` constraint
requiring the start time to be earlier than the referred time.
Finally, it uses
`ReviseConstraint` to find an existing root program, and revises
a sub-computation of it by the new constraint.
See Section 3, Section 4 of
[the paper](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00333)
for more details about `refer` and `ReviseConstraint`.

It is worth noting that the above program is a valid revision program
regardless of the type of the previous request: whether it is a creation,
query, update, deletion or another revision,
they all operate on a
`Event` constraint. This uniformity guarantees that the program
can be written independent of the dialogue history.


## API Calls and Model Operations

Some SMCalFlow library functions call web APIs or perform inference with a
trained model.
These do the bulk of the work and take up the bulk of the execution time.
The following is a complete list of the slow API calls and model operations:
 * RecipientAvailability,
 * FindReports,
 * FindManager,
 * UpdatePreflightEventWrapper,
 * CreatePreflightEventWrapper,
 * DeletePreflightEventWrapper,
 * FindEventWrapperWithDefaults,
 * RecipientWithNameLike,
 * Yield (generation model).

Some API calls also do real work, but are near-instantaneous:
 * DeleteCommitEventWrapper,
 * UpdateCommitEventWrapper,
 * CreateCommitEventWrapper,
 * EventAttendance,
 * refer (refer model),
 * ReviseConstraint (revise model).

The remainder of the functions are near-instantaneous.
They serve to construct the constraints and other arguments to the API calls
and model operations.
