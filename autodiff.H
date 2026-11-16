#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <compare>

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

    friend Dual abs (const Dual& x) {
        if (x.value < T{}) {
            return {-x.value, -x.deriv};
        } else if (x.value > T{}) {
            return x;
        }
        return {T{}, T{}};
    }

    // note, since our constructor is not explicit, these operators
    // work for Dual + Dual, Dual + T, or T + Dual

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


#endif
