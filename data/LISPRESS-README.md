# Lispress

*Lispress* is a lisp-like serialization format for programs.
It is intended to be human-readable, easy to work with in Python, and easy to
tokenize and predict with a standard seq2seq model.


Here is an example program in Lispress (a response to the utterance
`"what is my appointment with janice kang"`):
```clojure
(yield
  (:id
    (singleton
      (:results
        (FindEventWrapperWithDefaults
          :constraint (StructConstraint[Event]
            :attendees (AttendeeListHasRecipientConstraint
              :recipientConstraint (RecipientWithNameLike
                :constraint (StructConstraint[Recipient])
                :name #(PersonName "janice kang")))))))))
```


## Syntax

A Lispress program is an s-expression: either
a bare symbol, or
a whitespace-separated list of s-expressions, surrounded by parentheses.

### Values

Value literals are represented with a hash character followed by an
s-expression containing the name of the schema (i.e. type) of the data, followed by a
json-encoded string literal of the data surrounded by double-quotes.
For example: `#(PersonName "janice kang")`.
A `Number` may omit the double-quotes, e.g. `#(Number 4)`.

### Function application

The most common form in Lispress is a function applied to zero or more
arguments.
Function application expressions are lists,
with the first element of the list denoting the function,
and the remainder of the elements denoting its arguments.
There are two kinds of function application:

#### Named arguments
If the name of a function begins with a capitalized letter (`[A-Z]`),
then it accepts named arguments (and only named arguments).
The name of each named argument is prefixed with a colon character,
and named arguments are written after the function as alternating
`:name value` pairs.
Named arguments can be given in any order (when rendering, we alphabetize named arguments).

For example, in
```clojure
(DateAtTimeWithDefaults
  :date (Tomorrow)
  :time (NumberAM :number #(Number 10))
```
the `DateAtTimeWithDefaults` function is a applied to two named arguments.
`(Tomorrow)` is passed to the function as the `date` argument, and
`(NumberAM :number #(Number 10)` is passed in as the `time` argument.
`(Tomorrow)` is an example of a function applied to zero named arguments.
Some functions accepting named arguments may not require all arguments to be present.
You will often see the `StructConstraint[Event]` function being called without
a `:subject` or an `:end`, for example.

#### Positional arguments
If the name of a function does not begin with a capitalized letter
(i.e. it is lowercase or symbolic), then it accepts positional
arguments (and only positional arguments).
For example,
```clojure
(?= #(String "soccer game"))
```
represents the function `?=` being
applied to the single argument `#(String "soccer game")`.
And `(toDays #(Number 10))` is the function `toDays` applied to the single
argument `#(Number 10)`.


### Sugared `get`

There is a common construct in our programs where the `get` function
retrieves a field (specified by a `Path`) from a structured object.
For example,
```clojure
(get
  (refer (StructConstraint[Event]))
  #(Path "attendees"))
```
returns the `attendees` field of the salient `Event`.
When the path is a valid identifier (i.e. contains no whitespace or special
characters), the following sugared version is equivalent and preferred:
```clojure
(:attendees
  (refer (StructConstraint[Event])))
```




### Variable binding with `let`

To use a value more than once, it can be given a variable name using a `let`
binding.
A `let` binding is a list with three elements,
- the keyword `let`,
- a "binding" list containing alternating `variableName variableValue` pairs, and
- a program body, in which variable names bound in the previous form can be
referenced.

For example, in the following response to `"Can you find some past events on my calendar?"`,
```clojure
(let
  (x0 (Now))
  (yield
    (FindEventWrapperWithDefaults
      :constraint (EventOnDateBeforeTime
        :date (:date x0)
        :event (StructConstraint[Event])
        :time (:time x0)))))
```
the variable `x0` is assigned the value `(Now)` and then used twice in the body.
Note that `(Now)` is only evaluated once.
`let` bindings are an important mechanism to reuse the result of a
side-effecting computation.
For example, depending on the implementation of `Now`, the
following program may be referencing different values in the `:date` and `:time` fields:
```clojure
(FindEventWrapperWithDefaults
  :constraint (EventOnDateBeforeTime
    :date (:date (Now))
    :event (StructConstraint[Event])
    :time (:time (Now)))))
```

### Performing multiple actions in a turn with `do`

Two or more statements can be sequenced using the `do` keyword.
Each statement in a `do` form is fully interpreted and executed before any following
statements are.
In
```clojure
(do
  (ConfirmAndReturnAction)
  (yield
    (:start
      (FindNumNextEvent
        :constraint (StructConstraint[Event])
        :number #(Number 1)))))
```
for example, `ConfirmAndReturnAction` is guaranteed to execute before `FindNumNextEvent`.




## Code

Code for parsing and rendering Lispress is in the `dataflow.core.lispress`
package.

`parse_lispress` converts a string into a `Lispress` object, which is a nested
list-of-lists with `str`s as leaves.
`render_compact` renders `Lispress` on a single line (used in our `jsonl` data
files), and `render_pretty` renders with indentation, which is easier to read.

`lispress_to_program` and `program_to_lispress` convert to and from a `Program` object,
which is closer to a computation DAG (rather than an abstract syntax tree), and
is sometimes more convenient to work with.
