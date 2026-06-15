# SMIT Class Notes - JavaScript Fundamentals

## Topic: Variables (var, let, const)
- `const` is for values that never change after assignment.
- `let` is for values that will change.
- Avoid `var` — it is function-scoped and causes hoisting bugs.

## Topic: Functions
- Prefer small, single-purpose functions.
- Always name functions with a verb describing what they do: `calculateTotal()`, `getUserName()`.
- Avoid duplicating logic — extract repeated code into a function.

## Topic: Loops
- `for` loops are used when the number of iterations is known.
- `while` loops are used when the condition depends on runtime values.
- Always make sure the loop variable is updated, or it becomes an infinite loop.

## Topic: Naming Conventions
- Variables and functions: camelCase (`studentName`, `totalMarks`).
- Files: lowercase with hyphens or no spaces (`index.js`, `calculator.js`).
- Constants that never change: UPPER_SNAKE_CASE (`MAX_STUDENTS`).

## Topic: DOM Manipulation
- Use `document.querySelector()` or `getElementById()` to select elements.
- Use `addEventListener()` instead of inline `onclick`.
- Always check if an element exists before manipulating it to avoid `null` errors.

## Topic: Code Quality
- Use `===` and `!==` for comparisons (avoid `==`/`!=`).
- Remove leftover `console.log()` before final submission.
- Keep consistent indentation (2 spaces recommended).
- Add short comments explaining *why*, not just *what*.

## Grading Philosophy (SMIT)
- Marks are split between: correctness (does it run and produce the right output),
  code quality (naming, structure, formatting), and best practices (no var, ===, modular functions).
- Partial marks are given for logic that is mostly correct but has minor bugs.
- Zero marks for plagiarized/copy-pasted assignments from classmates.
