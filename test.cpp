#include <iostream>

#include "autodiff.H"


template <typename T>
T func(T x) {
    return 2 * x * x + 3 * x + 4;
}

int main() {

    Dual x(3.0);
    x.seed();

    auto f = func(x);

    std::cout << f.value << " " << f.deriv << std::endl;

}
