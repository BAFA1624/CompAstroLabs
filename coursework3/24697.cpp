#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

// Stringify things :)
#define STR( s ) #s

// Define some helper functions so arrays can be:
//  - Memberwise addition/subtraction by other arrays and floating point values.
//  - Memberwise multiplication/division by floating point values.

// -----------------------------------------------------------------------------//
// This isn't strictly necessary, I just think it's REALLY cool.
// Some template metaprogramming for efficiency
// When an array, e.g std::array<float, 10>, is +/-/etc, the compiler will
// recursively unroll the memberwise operation so that a for loop can be avoided
// which adds "jmp" calls in the generated machine code. This all happens at
// compile time, rather than at runtime. A small improvement in most cases but
// sometimes worthwhile by allowing the comiler to make further optimizations.
// -----------------------------------------------------------------------------//
template <typename T>
using trivial_op = std::function<void( T &, const T )>;

// Base cases for template metaprogramming recursive magic
template <typename T, std::size_t Size>
void
memberwise_op( [[maybe_unused]] const trivial_op<T> &       op,
               [[maybe_unused]] std::array<T, Size> &       lhs,
               [[maybe_unused]] const std::array<T, Size> & rhs,
               [[maybe_unused]] std::integral_constant<std::size_t, 0> ) {}
template <typename T, std::size_t Size>
void
memberwise_op( [[maybe_unused]] const trivial_op<T> & op,
               [[maybe_unused]] std::array<T, Size> & lhs,
               [[maybe_unused]] const T               rhs,
               [[maybe_unused]] std::integral_constant<std::size_t, 0> ) {}

template <typename T, std::size_t Size, std::size_t I>
void
memberwise_op( const trivial_op<T> & op, std::array<T, Size> & lhs,
               const std::array<T, Size> & rhs,
               std::integral_constant<std::size_t, I> ) {
    op( lhs[I - 1], rhs[I - 1] );
    memberwise_op( op, lhs, rhs, std::integral_constant<std::size_t, I - 1>() );
}
template <typename T, std::size_t Size, std::size_t I>
void
memberwise_op( const trivial_op<T> & op, std::array<T, Size> & lhs, const T rhs,
               std::integral_constant<std::size_t, I> ) {
    op( lhs[I - 1], rhs );
    memberwise_op( op, lhs, rhs, std::integral_constant<std::size_t, I - 1>() );
}

// Some macros to automatically define operator overloads for array + array,
// array + var, var * array, etc.
#define OP_FUNC( op ) []( auto & l, const auto r ) { l op r; }
#define APPLY_MEMBERWISE( op, lhs, rhs, Size )                    \
    memberwise_op( op, lhs, rhs,                                  \
                   std::integral_constant<std::size_t, Size>() ); \
    return lhs;

#define REF_OP_ARR_ARR( op )                                           \
    template <typename T, std::size_t Size>                            \
    inline std::array<T, Size> & operator op(                          \
        std::array<T, Size> & lhs, const std::array<T, Size> & rhs ) { \
        const std::function<void( T &, const T )> f = OP_FUNC( op );   \
        APPLY_MEMBERWISE( f, lhs, rhs, Size );                         \
    }
#define REF_OP_ARR_CONST( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> & operator op( std::array<T, Size> & lhs,       \
                                              const T               rhs ) {                  \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op r;                                                            \
        };                                                                     \
        memberwise_op( f, lhs, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return lhs;                                                            \
    }
#define REF_OP_CONST_ARR( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> & operator op( const T               lhs,       \
                                              std::array<T, Size> & rhs ) {    \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op r;                                                            \
        };                                                                     \
        memberwise_op( f, lhs, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return lhs;                                                            \
    }
#define REF_OP( op )       \
    REF_OP_ARR_ARR( op )   \
    REF_OP_ARR_CONST( op ) \
    REF_OP_CONST_ARR( op )

#define VAL_OP_ARR_ARR( op )                                                   \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> operator op(                                    \
        const std::array<T, Size> & lhs, const std::array<T, Size> & rhs ) {   \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op## = r;                                                        \
        };                                                                     \
        std::array<T, Size> tmp{ lhs };                                        \
        memberwise_op( f, tmp, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return tmp;                                                            \
    }
#define VAL_OP_ARR_CONST( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> operator op( const std::array<T, Size> & lhs,   \
                                            const T                     rhs ) {                    \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op## = r;                                                        \
        };                                                                     \
        std::array<T, Size> tmp{ lhs };                                        \
        memberwise_op( f, tmp, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return tmp;                                                            \
    }
#define VAL_OP_CONST_ARR( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> operator op(                                    \
        const T lhs, const std::array<T, Size> & rhs ) {                       \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op## = r;                                                        \
        };                                                                     \
        std::array<T, Size> tmp{ rhs };                                        \
        memberwise_op( f, tmp, lhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return tmp;                                                            \
    }
#define VAL_OP( op )       \
    VAL_OP_ARR_ARR( op )   \
    VAL_OP_ARR_CONST( op ) \
    VAL_OP_CONST_ARR( op )

// Required std::array operators
// clang-format off
REF_OP( += )
REF_OP( -= )
REF_OP( *= )
REF_OP( /= )
VAL_OP( + )
VAL_OP( - )
VAL_OP( * )
VAL_OP( / )
// clang-format on

// Define "state" & "flux" as fixed size (default = 3) arrays.
template <std::floating_point T, std::size_t Size = 3>
using state = std::array<T, Size>;
template <std::floating_point T, std::size_t Size = 3>
using flux = state<T, Size>;


// Class enum for selecting between type of algorithm.
// The value of each enum name is set to the required no. of ghost cells
enum class solution_type { lax_friedrichs = 1 };

// Fluid dynamics solver definition
template <std::floating_point T, std::size_t Size, solution_type Type>
class fluid_solver
{
    private:
    // An array of state arrays (3 values) with length Size + 2 * no. of ghost
    // cells each side
    std::array<state<T>, Size + 2 * static_cast<std::size_t>( Type )> m_state;
};

template <std::floating_point T, std::size_t Size>
void
print_array( const std::array<T, Size> & a ) {
    std::string s{ "array: { " };
    for ( std::uint64_t i{ 0 }; i < Size; ++i ) {
        s += std::to_string( a[i] ) + ", ";
    }
    s.pop_back();
    s.pop_back();
    s += " }";
    std::cout << s << std::endl;
}

int
main() {
    std::array<double, 3>       a1{ 1, 2, 3 };
    const std::array<double, 3> a2{ 1, 2, 3 };

    a1 += a2;
    print_array( a1 );

    a1 -= 3.;
    print_array( a1 );

    a1 *= 3.;
    print_array( a1 );

    state<double>       s1{ 1, 2, 3 };
    const state<double> s2{ 1, 2, 3 };

    s1 += s2;
    print_array( s1 );
    std::cout << std::endl;
    const auto s3 = s1 + s2;
    print_array( s3 );
}