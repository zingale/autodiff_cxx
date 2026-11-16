#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <cassert>
#include <cmath>
#include <compare>
#include <type_traits>

namespace dual {

    ///
    /// A simple Dual class containing a value and a derivative.
    ///
    template <typename T>
    class Dual {

    public:

        T value{};
        T deriv{};

        Dual (T _value, T _deriv=T{})
          : value{_value}, deriv{_deriv} {}

        void seed() {
            deriv = 1.0;
        }

        // note, since our constructor is not explicit, these operators
        // work for Dual + Dual, Dual + T, or T + Dual.  The ideal is that
        // if we do Dual + double, then the double will first be cast to
        // a Dual.

        friend auto operator<=> (const Dual& lhs, const Dual& rhs) {
            return lhs.value <=> rhs.value;
        }

        friend bool operator== (const Dual& lhs, const Dual& rhs) {
            return lhs.value == rhs.value;
        }

        friend Dual operator+ (const Dual& lhs, const Dual& rhs) {
            return {lhs.value + rhs.value,
                    lhs.deriv + rhs.deriv};
        }

        friend Dual operator- (const Dual& lhs, const Dual& rhs) {
            return {lhs.value - rhs.value,
                    lhs.deriv - rhs.deriv};
        }

        Dual operator- () const {return {-value, -deriv};}

        friend Dual operator* (const Dual& lhs, const Dual& rhs) {
            return {lhs.value * rhs.value,
                    lhs.value * rhs.deriv + lhs.deriv * rhs.value};
        }

        friend Dual operator/ (const Dual& lhs, const Dual& rhs) {
            return {lhs.value / rhs.value,
                    -lhs.value / (rhs.value * rhs.value) * rhs.deriv +
                    lhs.deriv / rhs.value};
        }

    };

    // A constant exponent does not need the logarithm used for a Dual exponent.
    template <typename T, typename U>
        requires (std::is_floating_point_v<U> || std::is_integral_v<U>)
    Dual<T> pow (const Dual<T>& x, U y) {

        if (y == U{0}) {
            return {T{1}, T{0}};
        }
        if (y == U{1}) {
            return x;
        }

        // Subtract in floating point to avoid overflowing a minimum integer.
        using Exponent = std::conditional_t<std::is_integral_v<U>, long double, U>;
        auto exponent = static_cast<Exponent>(y);
        T value = std::pow(x.value, exponent);
        T deriv = exponent * std::pow(x.value, exponent - Exponent{1}) * x.deriv;
        return {value, deriv};

    }

    // Deduce T from x, allowing convertible exponents as well as Dual<T>.
    template <typename T>
    Dual<T> pow (const Dual<T>& x, const std::type_identity_t<Dual<T>>& y) {

        if (y.deriv == T{}) {
            return pow(x, y.value);
        }

        // At zero with a positive exponent, the exponent partial is zero
        // (the limit from positive bases). Use the constant-exponent partial.
        if (x.value == T{} && y.value > T{}) {
            return pow(x, y.value);
        }

        // A varying exponent otherwise requires a strictly positive base.
        assert (x.value > T{});

        if (y.value == T{}) {
            return {T{1}, y.deriv * std::log(x.value)};
        }

        T value = std::pow(x.value, y.value);
        T deriv = value * (y.deriv * std::log(x.value) +
                           y.value * x.deriv / x.value);
        return {value, deriv};

    }

    template <typename T>
    Dual<T> abs (const Dual<T>& x) {
        if (std::isnan(x.value)) {
            return x;
        }
        if (x.value < T{}) {
            return {-x.value, -x.deriv};
        } else if (x.value > T{}) {
            return x;
        }
        return {T{}, T{}};
    }

    template <typename T>
    Dual<T> exp (const Dual<T>& x) {
        auto value = std::exp(x.value);
        return {value, value * x.deriv};
    }

    template <typename T>
    Dual<T> log (const Dual<T>& x) {
        auto value = std::log(x.value);
        return {value, x.deriv / x.value};
    }

    template <typename T>
    Dual<T> log1p (const Dual<T>& x) {
        auto value = std::log1p(x.value);
        return {value, x.deriv / (T{1} + x.value)};
    }

    template <typename T>
    Dual<T> sqrt (const Dual<T>& x) {
        auto value = std::sqrt(x.value);
        return {value, T{0.5} * x.deriv / value};
    }

    template <typename T>
    Dual<T> cbrt (const Dual<T>& x) {
        auto value = std::cbrt(x.value);
        return {value, (T{1} / T{3}) * x.deriv / (value * value)};
    }

    template <typename T>
    Dual<T> cos (const Dual<T>& x) {
        return {std::cos(x.value), -std::sin(x.value) * x.deriv};
    }

    template <typename T>
    Dual<T> sin (const Dual<T>& x) {
        return {std::sin(x.value), std::cos(x.value) * x.deriv};
    }

    template <typename T>
    Dual<T> atan (const Dual<T>& x) {
        return {std::atan(x.value), x.deriv / (T{1} + x.value * x.value)};
    }

}

#endif
