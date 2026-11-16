# `autodiff_cxx`

A very simple forward-mode autodiff using C++20.  It supports a derivative
pair (for two-independent variables).

This is header-only, just include `autodiff.H`.  See `test.cpp` for
examples.

`Dual<T, D>` stores a scalar value of type `T` and derivatives of type
`D`.  `D` defaults to `T`, so `dual::Dual<double>` carries one
derivative as before.

For two independent variables, use
`dual::DerivativePair<T>`:

```cpp
dual::Dual x(1.3, dual::DerivativePair(1.0, 0.0));
dual::Dual y(2.1, dual::DerivativePair(0.0, 1.0));

auto f = dual::exp(x * y) + dual::pow(x, y);
// f.value is the function value.
// f.deriv.d1 is the partial derivative with respect to x.
// f.deriv.d2 is the partial derivative with respect to y.
```

The explicit type for these variables is `dual::Dual<double,
dual::DerivativePair<double>>`. Use the same floating-point type for
values and pair components, and the same derivative type for all `Dual`
operands in an expression. Pairs support addition, subtraction,
negation, scalar multiplication on either side, and division by a
scalar.

All arithmetic and math helpers propagate both components. Comparisons
use only the value. `dual::min` and `dual::max` accept scalar bounds
on either side; selecting a bound gives zero derivatives, and ties
select the first argument.  `dual::constant_like(x, value)` constructs
a constant with the same value and derivative types as `x`.

Constructing a `Dual` without a seed gives zero derivatives. `seed()`
sets a scalar derivative to one, or a pair to `{1, 0}`. Use `seed({0,
1})` to select the second component. `pow(x, 0)` and `abs` at zero
return zero derivatives of the appropriate type.

Build and run the tests with:

```sh
g++ -std=c++20 -Wall -Wextra -pedantic-errors test.cpp -o test
./test
```
