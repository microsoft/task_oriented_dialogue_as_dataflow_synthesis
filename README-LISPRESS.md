# Lispress

*Lispress* is a lisp-like serialization format for programs.
It is intended to be human-readable, easy to work with in Python, and easy to
tokenize and predict with a standard seq2seq model. An older version, Lispress 1.0,
is described in [this README](README-LISPRESS-1.0.md). The current code is backwards
compatible with Lispress 1.0 programs. 


Here is an example program in Lispress (a response to the utterance
`"what is my appointment with janice kang"`):
```clojure
(Yield 
  (Event.id 
    (singleton 
      (QueryEventResponse.results 
        (FindEventWrapperWithDefaults 
          (Event.attendees? 
            (AttendeeListHasRecipientConstraint 
              (RecipientWithNameLike 
                (^(Recipient) EmptyStructConstraint) 
                (PersonName.apply "janice kang")))))))))
```


## Syntax

A Lispress program is an s-expression: either
a bare symbol, or
a whitespace-separated list of s-expressions, surrounded by parentheses. There is a little
bit of special syntax:
* Strings surrounded by double-quotes (`"`) are treated parsed as a single are symbol
  (including the quotes), with standard JSON escaping for strings. For example,
  ```clojure
  (MyFunc "this is a (quoted) string with a \" in it")
  ```
  will pass the symbol `"this is a (quoted) string with a \" in it"` to `MyFunc`. 
* The meta character (`^`) 
  ([borrowed from Clojure](https://clojure.org/reference/metadata))
  can be used for type ascriptions and type arguments. For example,
  ```clojure
  ^Number 1
  ```
  would be written as `1: Number` in Scala. A list marked by the meta character
  in the first argument of an s-expression is interpreted as a list of type arguments.
  For example,
  ```clojure
  (^(Number) MyFunc 1)
  ```
  would be written as `MyFunc[Number](1)` in Scala or `MyFunc<Number>(1)` in Swift and Rust.
* (Deprecated) The reader macro character (`#`), 
  [borrowed from Common Lisp](https://gist.github.com/chaitanyagupta/9324402) 
  marks literal values.
  For example, `#(PersonName "John")` marks a value of type `PersonName` with 
  content `"John"`. Reader macros are no longer in Lispress 2.0. Instead, 
  standard literals like booleans, longs, numbers, and strings, can be written directly,
  while wrapper types (like `PersonName`) feature an explicit call to a constructor
  like `PersonName.apply`. The current code will interpret Lispress 1.0
  `Number`s and `String`s as their bare equivalents, so `#(String "foo")` and `"foo"`
  will be interpreted as the same program. Similarly, `#(Number 1)` and `1` will
  be interpreted as the same program.
* Literals of type Long are written as an integer literal followed by an `L` (e.g. `12L`) 
  as in Java/Scala.

### Function application

The most common form in Lispress is a function applied to zero or more
arguments.
Function application expressions are lists,
with the first element of the list denoting the function,
and the remainder of the elements denoting its arguments.
We follow Common Lisp and Clojure in using `:` to denote named arguments. For example,
`(MyFunc :foo 1)` would be `MyFunc(foo = 1)` in Scala or Python. At present, functions 
must either be entirely positional or entirely named, and only functions with an
uppercase letter for the first character may take named arguments. 

### (Deprecated) Sugared `get`

There is a common construct in the SMCalFLow 1.x dataset where the `get` function
retrieves a field (specified by a `Path`) from a structured object.
For example,
```clojure
(get
  (refer (StructConstraint[Event]))
  #(Path "attendees"))
```
returns the `attendees` field of the salient `Event`.
For backwards compatibility with Lispress 1.0, the parser will accept
the following equivalent form. 
```clojure
(:attendees (refer (StructConstraint[Event])))
```

In updated Lispress, accessor functions contain the name of the type they access:
```clojure
(Event.attendees (refer (^(Event) StructConstraint)))
```

`Path`s have been removed completely from SMCalflow 2.0.




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
  (Yield 
    (FindEventWrapperWithDefaults 
      (EventOnDateBeforeTime 
        (DateTime.date x0) 
        (^(Event) EmptyStructConstraint) 
        (DateTime.time x0)))))
```
the variable `x0` is assigned the value `(Now)` and then used twice in the body.
Note that `(Now)` is only evaluated once.
`let` bindings are an important mechanism to reuse the result of a
side-effecting computation.
For example, depending on the implementation of `Now`, the
following program may be produce different values in the `:date` and `:time` fields:
```clojure
(FindEventWrapperWithDefaults 
  (EventOnDateBeforeTime 
    (DateTime.date (Now))) 
    (^(Event) EmptyStructConstraint) 
    (DateTime.time (Now)))))
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
    (Event.start
      (FindNumNextEvent
        (^(Event) StructConstraint)
        1L))))
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
