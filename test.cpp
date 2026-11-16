#include <iostream>
#include <cassert>
#include <limits>

#include "autodiff.H"

// a simple helper function for comparing with roundoff

bool approx(double x, double y) {
    return std::abs(x - y) < 100 * std::numeric_limits<double>::epsilon();
}


int main() {

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

        assert (approx(f.value, 5.0 + 11.0 * std::exp(1.0)));
        assert (approx(f.deriv, -13.0 * std::exp(1.0)));

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
        assert (approx(variable.deriv, 8.0 * (std::log(2.0) + 1.5)));

    }

    {

        // A zero-valued exponent can still vary.
        auto result = dual::pow(dual::Dual(2.0, 3.0), dual::Dual(0.0, 2.0));
        assert (result.value == 1.0);
        assert (approx(result.deriv, 2.0 * std::log(2.0)));

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
