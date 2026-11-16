#include <iostream>
#include <cassert>
#include <limits>

#include "autodiff.H"

// Scale the roundoff tolerance with both the precision and result
// magnitude.

template <typename T>
void close(T actual, T expected) {
    assert (std::abs(actual - expected) <=
            100 * std::numeric_limits<T>::epsilon() * (T{1} + std::abs(expected)));
}


// Verify two independent derivatives for each floating-point type
// used below.

template <typename T>
void test_derivative_pair() {

    using Pair = dual::DerivativePair<T>;
    using Dual = dual::Dual<T, Pair>;

    {
        // Pair arithmetic acts component by component; scalars scale both
        // entries.

        Pair a{T{2}, T{-3}};
        Pair b{T{-1}, T{4}};

        assert ((a + b == Pair{T{1}, T{1}}));
        assert ((a - b == Pair{T{3}, T{-7}}));
        assert ((-a == Pair{T{-2}, T{3}}));

        assert ((a * 2 == Pair{T{4}, T{-6}}));
        assert ((2 * a == Pair{T{4}, T{-6}}));
        assert ((a / 2 == Pair{T{1}, T{-1.5}}));
    }

    {
        // Create a function f(x, y) and check the partials

        T xv = static_cast<T>(1.3);
        T yv = static_cast<T>(2.1);

        // Seed x and y independently so
        //   d1 = partial/partial x
        //   d2 = partial/partial y

        dual::Dual x(xv, dual::DerivativePair(T{1}, T{0}));
        dual::Dual y(yv, dual::DerivativePair(T{0}, T{1}));

        static_assert(std::is_same_v<decltype(x), Dual>);

        auto f = dual::exp(x * y) + x / y - 3 / x + dual::pow(x, y) - y + 2 * x;

        close(f.value, std::exp(xv * yv) + xv / yv - 3 / xv + std::pow(xv, yv) - yv + 2 * xv);

        // Compare each propagated component with its analytic partial derivative.

        close(f.deriv.d1, yv * std::exp(xv * yv) + 1 / yv + 3 / (xv * xv) +
                          yv * std::pow(xv, yv - T{1}) + 2);

        close(f.deriv.d2, xv * std::exp(xv * yv) - xv / (yv * yv) +
                          std::pow(xv, yv) * std::log(xv) - 1);
    }

    {

        // Think of z as z(u, v), with its partial derivatives stored in z.deriv.
        // For each helper, the chain rule multiplies its slope by these seeds.

        T xv = static_cast<T>(1.3);
        Pair a{T{2}, T{-3}};

        Dual z(xv, a);

        auto check = [&z] (const Dual& result, T value, T slope) {
            close(result.value, value);
            close(result.deriv.d1, z.deriv.d1 * slope);
            close(result.deriv.d2, z.deriv.d2 * slope);
        };

        check(dual::exp(z), std::exp(xv), std::exp(xv));

        check(dual::log(z), std::log(xv), 1 / xv);
        check(dual::log1p(z), std::log1p(xv), 1 / (1 + xv));

        check(dual::sqrt(z), std::sqrt(xv), T{0.5} / std::sqrt(xv));
        check(dual::cbrt(z), std::cbrt(xv), (T{1} / T{3}) / (std::cbrt(xv) * std::cbrt(xv)));

        check(dual::cos(z), std::cos(xv), -std::sin(xv));
        check(dual::sin(z), std::sin(xv), std::cos(xv));
        check(dual::atan(z), std::atan(xv), 1 / (1 + xv * xv));

        check(dual::pow(z, 3), std::pow(xv, 3), 3 * xv * xv);

        check(-z, -xv, T{-1});

        check(dual::abs(-z), xv, T{1});
        check(dual::abs(z), xv, T{1});
    }

    {

        // check constants and special cases

        T xv = static_cast<T>(1.3);
        Pair a{T{2}, T{-3}};
        Pair b{T{-1}, T{4}};
        Dual z(xv, a);

        // Constant expressions preserve the pair type but clear both
        // derivatives.

        auto constant = dual::constant_like(z, 7);
        assert (constant.value == T{7} && constant.deriv == Pair{});

        auto zeroth = dual::pow(z, 0);
        assert (zeroth.value == T{1} && zeroth.deriv == Pair{});

        // abs uses a zero-derivative convention at its
        // nondifferentiable origin.

        auto origin = dual::abs(Dual(T{0}, a));
        assert (origin.value == T{0} && origin.deriv == Pair{});

        // An invalid value must remain NaN rather than falling
        // through to zero.

        assert (std::isnan(dual::abs(Dual(std::numeric_limits<T>::quiet_NaN(), a)).value));

        // Selecting a scalar bound gives a locally constant result
        // with zero seeds.

        auto lower = dual::min(z, -1);
        auto upper = dual::max(3, z);

        assert (lower.value == T{-1} && lower.deriv == Pair{});
        assert (upper.value == T{3} && upper.deriv == Pair{});

        // Selecting the Dual preserves its derivatives, with scalars
        // on either side.

        assert (dual::min(3, z).deriv == a);
        assert (dual::max(z, -1).deriv == a);

        // Ties select the first argument, including its derivatives
        // (or zero seeds).

        assert (dual::min(z, Dual(xv, b)).deriv == a);
        assert (dual::max(Dual(xv, b), z).deriv == b);
        assert (dual::min(z, xv).deriv == a);
        assert (dual::max(xv, z).deriv == Pair{});

        // Comparisons depend only on values, independent of the
        // derivative seeds.

        assert (z == Dual(xv, b));
        assert (z > 0 && 0 < z);

        // At exponent value zero, a varying exponent still
        // contributes log(base) * b.

        auto exponent_zero = dual::pow(Dual(T{2}, a), Dual(T{0}, b));
        close(exponent_zero.deriv.d1, -std::log(T{2}));
        close(exponent_zero.deriv.d2, 4 * std::log(T{2}));

        // At base zero and exponent two, both partial derivatives vanish.

        auto zero_base = dual::pow(Dual(T{0}, a), Dual(T{2}, b));
        assert (zero_base.value == T{0} && zero_base.deriv == Pair{});

        // A zero-seeded Dual exponent uses the constant-exponent rule
        // for negative bases.

        auto fixed_exponent = dual::pow(Dual(T{-2}, a), Dual(T{3}));
        assert ((fixed_exponent.deriv == Pair{T{24}, T{-36}}));

        // Forming exponent - 1 must not overflow for the minimum signed integer.

        auto minimum = dual::pow(Dual(T{1}, a), std::numeric_limits<int>::min());
        close(minimum.deriv.d1, 2 * static_cast<T>(std::numeric_limits<int>::min()));
        close(minimum.deriv.d2, -3 * static_cast<T>(std::numeric_limits<int>::min()));

        // Construction starts with zero seeds; seed() selects d1 by default,
        // and an explicit pair can instead select the second independent variable.

        Dual seeded(T{2});
        assert (seeded.deriv == Pair{});
        seeded.seed();
        assert ((seeded.deriv == Pair{T{1}, T{0}}));
        seeded.seed({T{0}, T{1}});
        assert ((seeded.deriv == Pair{T{0}, T{1}}));
}


int main() {

    test_derivative_pair<float>();
    test_derivative_pair<double>();
    test_derivative_pair<long double>();

    {

        // basic addition

        auto func = [] (dual::Dual<double> x) -> dual::Dual<double> {return x + 2;};

        dual::Dual x(2.0);
        x.seed();

        auto f = func(x);

        assert (f.value == 4.0);
        assert (f.deriv == 1.0);


        auto func2 = [] (dual::Dual<double> x) -> dual::Dual<double> {return 2 + x;};

        dual::Dual y(2.0);
        y.seed();

        auto f2 = func2(y);

        assert (f2.value == 4.0);
        assert (f2.deriv == 1.0);

    }

    {

        // a simple function with multiplications and addition

        auto func = [] (dual::Dual<double> x) -> dual::Dual<double>
            {return 2 * x * x + 3 * x + 4;};

        dual::Dual x(3.0);
        x.seed();

        auto f = func(x);

        assert (f.value == 31.0);
        assert (f.deriv == 15.0);

    }

    {

        // test pow

        auto func = [] (dual::Dual<double> x) -> dual::Dual<double>
            {return 2 * dual::pow(x, 3.0);};

        dual::Dual x(2.5);
        x.seed();

        auto f = func(x);

        assert (f.value == 31.25);
        assert (f.deriv == 37.5);

    }

    {

        // test exp and division

        auto func = [] (dual::Dual<double> x) -> dual::Dual<double>
            {return (dual::pow(x, 2.0) + 10.0) / dual::exp(x) + 5.0;};

        dual::Dual x(-1.0);
        x.seed();

        auto f = func(x);

        close(f.value, 5.0 + 11.0 * std::exp(1.0));
        close(f.deriv, -13.0 * std::exp(1.0));

    }

    {

        // Constant exponents support negative and zero bases.
        dual::Dual x(-2.0, 2.0);
        auto square = dual::pow(x, 2.0f);
        assert (square.value == 4.0);
        assert (square.deriv == -8.0);

        auto cube = dual::pow(x, 3);
        assert (cube.value == -8.0);
        assert (cube.deriv == 24.0);

        auto inverse = dual::pow(x, -1);
        assert (inverse.value == -0.5);
        assert (inverse.deriv == -0.5);

        dual::Dual zero(0.0, 2.0);
        auto constant = dual::pow(zero, 0.0);
        auto linear = dual::pow(zero, 1.0f);
        auto quadratic = dual::pow(zero, 2.0);
        assert (constant.value == 1.0 && constant.deriv == 0.0);
        assert (linear.value == 0.0 && linear.deriv == 2.0);
        assert (quadratic.value == 0.0 && quadratic.deriv == 0.0);

        // Dual exponents still include the exponent's derivative.
        auto variable = dual::pow(dual::Dual(2.0, 1.0), dual::Dual(3.0, 1.0));
        assert (variable.value == 8.0);
        close(variable.deriv, 8.0 * (std::log(2.0) + 1.5));

    }

    {

        // A zero-valued exponent can still vary.
        auto result = dual::pow(dual::Dual(2.0, 3.0), dual::Dual(0.0, 2.0));
        assert (result.value == 1.0);
        close(result.deriv, 2.0 * std::log(2.0));

        dual::Dual zero(0.0, 2.0);
        auto constant = dual::pow(zero, dual::Dual(0.0));
        auto linear = dual::pow(zero, dual::Dual(1.0));
        auto square = dual::pow(zero, dual::Dual(2.0));
        auto varying = dual::pow(zero, dual::Dual(2.0, 1.0));
        assert (constant.value == 1.0 && constant.deriv == 0.0);
        assert (linear.value == 0.0 && linear.deriv == 2.0);
        assert (square.value == 0.0 && square.deriv == 0.0);
        assert (varying.value == 0.0 && varying.deriv == 0.0);

        auto minimum = dual::pow(dual::Dual(1.0, 1.0), std::numeric_limits<int>::min());
        assert (minimum.value == 1.0);
        assert (minimum.deriv == static_cast<double>(std::numeric_limits<int>::min()));

        auto nan = dual::abs(dual::Dual(std::numeric_limits<double>::quiet_NaN()));
        assert (std::isnan(nan.value));

    }

    {

        // Exercise float instantiations without narrowing conversions.
        dual::Dual x(1.0f, 2.0f);
        auto logarithm = dual::log1p(x);
        auto root = dual::sqrt(x);
        auto cube_root = dual::cbrt(x);
        auto angle = dual::atan(x);
        assert (logarithm.value == std::log1p(1.0f) && logarithm.deriv == 1.0f);
        assert (root.value == 1.0f && root.deriv == 1.0f);
        assert (cube_root.value == 1.0f && cube_root.deriv == 2.0f / 3.0f);
        assert (angle.value == std::atan(1.0f) && angle.deriv == 1.0f);

        auto power = dual::pow(x, dual::Dual(0.0f, 1.0f));
        assert (power.value == 1.0f && power.deriv == 0.0f);
        auto extended = dual::pow(dual::Dual(2.0L, 1.0L), dual::Dual(3.0L));
        assert (extended.value == 8.0L && extended.deriv == 12.0L);

    }

}
