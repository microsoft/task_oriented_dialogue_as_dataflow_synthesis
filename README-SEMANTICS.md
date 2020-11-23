# Understanding SMCalFlow programs

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
(Yield :output 
  (CreateCommitEventWrapper :event 
    (CreatePreflightEventWrapper :constraint 
      (Constraint[Event] 
        :start 
          (Constraint[DateTime]
            :date (?= (NextDOW :dow #(DayOfWeek \"MONDAY\")))
            :time (?= (NumberAM :number #(Number 11)))
          )) 
        :subject (?= #(String \"work meeting\"))
      )
    )
  )
)
```

This program constructs a compositional constraint of type `Constraint[Event]`,
which contains two sub-constraints: `start` and `subject`. Each sub-constraint
can be compositional too. For example, the `start` field (which represents
a constraint on the start time of the event) has two sub-constraints: `date` and
`time`.

Let's take a closer look at the `date` sub-field. This filed contains
a `Date` constraint constructed by the `?=` function.
Here, `?=` means "exact match". It takes an
reference object of type `T` and returns a constraint
that is satisfied if the input value is equal to the
reference. In this example, the `date` constraint is satisfied if
the event date is equal to "next Monday". Similarly, the `time` constraint
is satisfied if the event's start time is equal to 11 AM, and the `subject`
constraint is satisfied if the event's subject is equal to "work meeting".

Why do we want to use **constraints**, instead of concrete values, to
represent date, time and subject?
It is because that constraints are flexible enough to represent
the ambiguity in natural language. For example, if the user says "Create a work meeting
on Monday morning" or "Create a work meeting after the lunch",
she doesn't specify any concrete time, but both "morning" and
"after the lunch" can be represented as time constraints.

After the `Event` constraint is constructed, it is sent to
`CreatePreflightEventWrapper`, which solves a constraint satisfaction problem to figure out
the actual `Event` that the user wants to create. If a concrete solution
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

If the user says "Yes" in the next turn, then the confirmation program
will be like:
```
(Yield :output (Execute :intension (ConfirmAndReturnAction)))
```
The `ConfirmAndReturnAction` function puts a "confirm" token to the
dataflow graph and returns the last turn's program. The `Execute`
function then re-executes that program. This time, since the creation
is confirmed, the `CreateCommitEventWrapper` call will succeed and
the event will be added to the user's calendar.

## Query Event

The user can query events in the calendar. If the user says
"Where is the avocado festival?", then the program is:
```clojure
(Yield :output 
  (:location 
    (singleton 
      (:results 
        (FindEventWrapperWithDefaults :constraint 
          (Constraint[Event] :subject (?~= #(String \"avocado festival\")))
        )
      )
    )
  )
)
```
This program builds an `Event` constraint requiring the event's
subject to fuzzily match the string "avocado festival" (the `?~=`
function returns a "fuzzy match" constraint). It is then
sent to the `FindEventWrapperWithDefaults` for querying the database.
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
the user says "Chang my meeting tomorrow to 4 PM", then the program is:
```clojure
(Yield :output 
  (UpdateCommitEventWrapper :event 
    (UpdatePreflightEventWrapper 
      :id 
        (:id 
          (singleton 
            (:results 
              (FindEventWrapperWithDefaults :constraint
                (Constraint[Event] :start 
                  (Constraint[DateTime] :date (?= (Tomorrow)))) 
                )
              )
            )
          )
        )
      :update 
        (Constraint[Event] :start 
          (Constraint[DateTime] :time 
            (?= (NumberPM :number #(Number 4)))
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
(Yield :output 
  (DeleteCommitEventWrapper :event 
    (DeletePreflightEventWrapper :id 
      (:id 
        (singleton 
          (:results 
            (FindEventWrapperWithDefaults :constraint 
              (Constraint[Event] :attendees 
                (AttendeeListHasRecipientConstraint :recipientConstraint 
                  (RecipientWithNameLike :name #(PersonName \"Emma\"))
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
The program builds a `Constraint[Event]` specifying that the event's attendee
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
(Yield :output 
  (Execute :intension 
    (ReviseConstraint 
      :rootLocation (roleConstraint #(Path \"output\")) 
      :oldLocation (Constraint[Constraint[Event]]) 
      :new 
        (Constraint[Event] :start 
          (Constraint[DateTime] :time 
            (?< 
              (Execute :intension 
                (refer 
                  (andConstraint 
                    (roleConstraint 
                      (append #(List[Path] []) #(Path \"start\"))
                    ) 
                    (extensionConstraint (Constraint[Time]))
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
the start time of a previous event, then constructs a new `Constraint[Event]`
requiring the start time to be earlier than that time. Finally, it uses
`ReviseConstraint` to revise the `Constraint[Event]` in an existing
computation by the new constraint. See Section 3, Section 4 of
[the paper](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00333)
for more details about `refer` and `ReviseConstraint`.

It is worth noting that the above program is a valid revision program
regardless of the type of the previous request: no matter it is a creation,
query, update, deletion or another revision,
they are all performed based on a
`Constraint[Event]`. This uniformity guarantees that the program
can be written independent of the dialogue history.