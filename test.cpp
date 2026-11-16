#include <iostream>
#include <cassert>

#include "autodiff.H"


int main() {

    {

        // basic addition

        auto func = [] (Dual<double> x) -> Dual<double> {return x + 2;};

        Dual x(2.0);
        x.seed();

        auto f = func(x);

        assert (f.value == 4.0);
        assert (f.deriv == 1.0);


        auto func2 = [] (Dual<double> x) -> Dual<double> {return 2 + x;};

        Dual y(2.0);
        y.seed();

        auto f2 = func2(y);

        assert (f2.value == 4.0);
        assert (f2.deriv == 1.0);

    }

    {

        // a simple function with multiplications and addition

        auto func = [] (Dual<double> x) -> Dual<double> {return 2 * x * x + 3 * x + 4;};

        Dual x(3.0);
        x.seed();

        auto f = func(x);

        assert (f.value == 31.0);
        assert (f.deriv == 15.0);

    }

}
