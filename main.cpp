#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>

int main() {

  Kokkos::initialize();
  {

    double values[] = {1.0, 1.1, 1.12, 1.123, 1.1234, 1.12345, 1.123456, 1.1234567, 1.12345678, 1.123456789,
                       0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001,
                       10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000};

    using double_t = Kokkos::View<double[30]>;
    using float_t = Kokkos::View<float[30]>;

    auto A = double_t("double_view");
    auto B = float_t("float_view");

    auto h_A = Kokkos::create_mirror_view(A);
    auto h_B = Kokkos::create_mirror_view(B);

    for (int i=0; i<30; ++i)
    {
      h_A(i) = values[i];
      h_B(i) = values[i];
    }

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(B, h_B);
    Kokkos::fence();

    auto exp_A = double_t("exp_double");
    auto exp_B = float_t("exp_float");

    Kokkos::parallel_for("exp", 30, KOKKOS_LAMBDA (const int i)
    {
      exp_A(i) = exp( A(i) );
      exp_B(i) = exp( B(i) );
    });
    Kokkos::fence();

    Kokkos::deep_copy(h_A, exp_A);
    Kokkos::deep_copy(h_B, exp_B);
    Kokkos::fence();

    std::cout << "* * * EXP DOUBLE / FLOAT * * *" << std::endl;
    std::cout << "------------------------------" << std::endl;
    for (int i=0; i<30; ++i)
    {
      std::cout << std::setprecision(17) << "exp(" << values[i] << ")=" << h_A(i) << std::endl;
      std::cout << std::setprecision(17) << "exp(" << values[i] << ")=" << h_B(i) << std::endl;
    }

    auto sin_A = double_t("sin_double");
    auto sin_B = float_t("sin_float");
 
    Kokkos::parallel_for("sin", 30, KOKKOS_LAMBDA (const int i)
    {
      sin_A(i) = sin( A(i) );
      sin_B(i) = sin( B(i) );
    });
    Kokkos::fence();

    Kokkos::deep_copy(h_A, sin_A);
    Kokkos::deep_copy(h_B, sin_B);
    Kokkos::fence();

    std::cout << "* * * SIN DOUBLE / FLOAT * * *" << std::endl;
    std::cout << "------------------------------" << std::endl;
    for (int i=0; i<30; ++i)
    {
      std::cout << std::setprecision(17) << "sin(" << values[i] << ")=" << h_A(i) << std::endl;
      std::cout << std::setprecision(17) << "sin(" << values[i] << ")=" << h_B(i) << std::endl;
    }

  }
  Kokkos::finalize();

  return 0;
}
