#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


// Class enum for selecting between type of algorithm.
// The value of each enum name is set to the required no. of ghost cells
enum class solution_type : std::size_t {
    lax_friedrichs,
    lax_wendroff,
    hll,
    hllc
};
enum class boundary_type : std::size_t { outflow, reflecting };
enum class coordinate_type : std::size_t { cartesian, spherical };
enum class approx_order : std::size_t { first = 1, second = 2 };

const std::array<std::string, 4> solution_string{ "lax_friedrichs",
                                                  "lax_wendroff", "hll",
                                                  "hllc" };
const std::array<std::string, 2> approx_string{ "first_order", "second_order" };

// Stringify things :)
#define STR( s ) #s

// Define some helper functions so arrays can be:
//  - Memberwise addition/subtraction by other arrays and floating point values.
//  - Memberwise multiplication/division by floating point values.

// Some macros to automatically define operator overloads for array + array,
// array + var, var * array, etc.

#define REF_OP_ARR_ARR( op )                                            \
    template <typename T, std::size_t Size>                             \
    inline constexpr std::array<T, Size> & operator op(                 \
        std::array<T, Size> & lhs, const std::array<T, Size> & rhs ) {  \
        for ( std::size_t i{ 0 }; i < Size; ++i ) { lhs[i] op rhs[i]; } \
        return lhs;                                                     \
    }
#define REF_OP_ARR_ARR_3( op )                                   \
    template <typename T>                                        \
    inline constexpr std::array<T, 3> & operator op(             \
        std::array<T, 3> & lhs, const std::array<T, 3> & rhs ) { \
        lhs[0] op rhs[0];                                        \
        lhs[1] op rhs[1];                                        \
        lhs[2] op rhs[2];                                        \
        return lhs;                                              \
    }
#define REF_OP_ARR_CONST( op )                                       \
    template <typename T1, typename T2, std::size_t Size>            \
    inline constexpr std::array<T1, Size> & operator op(             \
        std::array<T1, Size> & lhs, const T2 rhs ) {                 \
        for ( std::size_t i{ 0 }; i < Size; ++i ) { lhs[i] op rhs; } \
        return lhs;                                                  \
    }
#define REF_OP_ARR_CONST_3( op )                                               \
    template <typename T1, typename T2>                                        \
    inline constexpr std::array<T1, 3> & operator op( std::array<T1, 3> & lhs, \
                                                      const T2 rhs ) {         \
        lhs[0] op rhs;                                                         \
        lhs[1] op rhs;                                                         \
        lhs[2] op rhs;                                                         \
        return lhs;                                                            \
    }
#define REF_OP_CONST_ARR( op )                                      \
    template <typename T1, typename T2, std::size_t Size>           \
    inline constexpr std::array<T2, Size> & operator op(            \
        const T1 lhs, std::array<T2, Size> & rhs ) {                \
        for ( std::size_t i = 0; i < Size; ++i ) { rhs[i] op lhs; } \
        return lhs;                                                 \
    }
#define REF_OP_CONST_ARR_3( op )                      \
    template <typename T1, typename T2>               \
    inline constexpr std::array<T2, 3> & operator op( \
        const T1 lhs, std::array<T2, 3> & rhs ) {     \
        rhs[0] op lhs;                                \
        rhs[1] op lhs;                                \
        rhs[2] op lhs;                                \
        return rhs;                                   \
    }
#define REF_OP( op )         \
    REF_OP_ARR_ARR_3( op )   \
    REF_OP_ARR_ARR( op )     \
    REF_OP_ARR_CONST_3( op ) \
    REF_OP_ARR_CONST( op )   \
    REF_OP_CONST_ARR_3( op ) \
    REF_OP_CONST_ARR( op )

#define VAL_OP_ARR_ARR( op )                                                 \
    template <typename T, std::size_t Size>                                  \
    inline constexpr std::array<T, Size> operator op(                        \
        const std::array<T, Size> & lhs, const std::array<T, Size> & rhs ) { \
        auto tmp{ lhs };                                                     \
        for ( std::size_t i{ 0 }; i < Size; ++i ) { tmp[i] op## = rhs[i]; }  \
        return tmp;                                                          \
    }
#define VAL_OP_ARR_ARR_3( op )                                         \
    template <typename T>                                              \
    inline constexpr std::array<T, 3> operator op(                     \
        const std::array<T, 3> & lhs, const std::array<T, 3> & rhs ) { \
        auto tmp{ lhs };                                               \
        tmp[0] op## = rhs[0];                                          \
        tmp[1] op## = rhs[1];                                          \
        tmp[2] op## = rhs[2];                                          \
        return tmp;                                                    \
    }
#define VAL_OP_ARR_CONST( op )                                                 \
    template <typename T1, typename T2, std::size_t Size>                      \
    inline std::array<T1, Size> operator op( const std::array<T1, Size> & lhs, \
                                             const T2 rhs ) {                  \
        auto tmp{ lhs };                                                       \
        for ( std::size_t i{ 0 }; i < Size; ++i ) { tmp[i] op## = rhs; }       \
        return tmp;                                                            \
    }
#define VAL_OP_ARR_CONST_3( op )                        \
    template <typename T1, typename T2>                 \
    inline constexpr std::array<T1, 3> operator op(     \
        const std::array<T1, 3> & lhs, const T2 rhs ) { \
        auto tmp{ lhs };                                \
        tmp[0] op## = rhs;                              \
        tmp[1] op## = rhs;                              \
        tmp[2] op## = rhs;                              \
        return tmp;                                     \
    }
#define VAL_OP_CONST_ARR( op )                                           \
    template <typename T1, typename T2, std::size_t Size>                \
    inline constexpr std::array<T2, Size> operator op(                   \
        const T1 lhs, const std::array<T2, Size> & rhs ) {               \
        auto tmp{ rhs };                                                 \
        for ( std::size_t i{ 0 }; i < Size; ++i ) { tmp[i] op## = lhs; } \
        return tmp;                                                      \
    }
#define VAL_OP_CONST_ARR_3( op )                        \
    template <typename T1, typename T2>                 \
    inline constexpr std::array<T2, 3> operator op(     \
        const T1 lhs, const std::array<T2, 3> & rhs ) { \
        auto tmp{ rhs };                                \
        tmp[0] op## = lhs;                              \
        tmp[1] op## = lhs;                              \
        tmp[2] op## = lhs;                              \
        return tmp;                                     \
    }
#define VAL_OP( op )         \
    VAL_OP_ARR_ARR_3( op )   \
    VAL_OP_ARR_ARR( op )     \
    VAL_OP_ARR_CONST_3( op ) \
    VAL_OP_ARR_CONST( op )   \
    VAL_OP_CONST_ARR_3( op ) \
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


template <typename T, std::size_t N, bool endpoint = true>
constexpr std::array<T, N>
linspace( const T a, const T b ) {
    const T dx{ ( b - a ) / static_cast<T>( endpoint ? N - 1 : N ) };

    std::array<T, N> arr{};
    for ( std::size_t i{ 0 }; i < N; ++i ) { arr[i] = a + i * dx; }

    return arr;
}

// Simple std::array printer
template <typename T, std::size_t Size>
std::string
array_string( const std::array<T, Size> & a ) {
    std::string s{ "array: { " };
    for ( std::uint64_t i{ 0 }; i < Size; ++i ) {
        s += std::to_string( a[i] ) + ", ";
    }
    s.pop_back();
    s.pop_back();
    s += " }";
    return s;
}

// Writes a set of arrays to file
template <typename T, std::size_t N>
void
write_to_file( const std::string &                   filename,
               const std::vector<std::array<T, N>> & values,
               const std::vector<std::string> &      keys = {} ) {
    assert( keys.size() == values.size() || keys.size() == 0 );

    // Find shortest array length, limits the number of rows written to file
    std::uint64_t n_rows{ std::numeric_limits<std::uint64_t>::max() };
    for ( const auto & v : values ) {
        n_rows = std::min<std::uint64_t>( n_rows, v.size() );
    }

    // Open file & write column titles if provided
    std::ofstream fp( filename );
    if ( keys.size() > 0 ) {
        std::string header{ "" };
        for ( const auto & key : keys ) { header += key + ","; }
        header.pop_back();
        header += "\n";
        fp << header.c_str();
    }

    // Write data
    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        std::string line{ "" };
        for ( const auto & v : values ) {
            line += std::to_string( v[i] ) + ",";
        }
        line.pop_back();
        line += "\n";
        fp << line.c_str();
    }
}

// Define "state" & "flux" as fixed size (default = 3) arrays.
template <typename T, std::size_t Size = 3>
using state = std::array<T, Size>;
template <typename T, std::size_t Size = 3>
using flux = state<T, Size>;

template <typename T>
constexpr inline T
sgn( const T x ) {
    return ( T( 0 ) < x ) - ( x < T( 0 ) );
}
template <typename T>
constexpr inline state<T>
sgn( const state<T> & x ) {
    return state<T>{ sgn( x[0] ), sgn( x[1] ), sgn( x[2] ) };
}

template <typename T>
constexpr state<T>
slope( const state<T> & Q1, const state<T> & Q2, const state<T> & Q3,
       const T dx, [[maybe_unused]] const T gamma ) {
    const auto a{ ( Q2 - Q1 ) / dx }, b{ ( Q3 - Q2 ) / dx },
        c{ ( Q3 - Q1 ) / dx };
    const state<T> sgn_a{ sgn( a ) };


    const auto slope_factor{ 0.25 * sgn_a * ( sgn_a + sgn( b ) )
                             * ( sgn_a + sgn( c ) ) };
    // clang-format off
    const auto slope{
        slope_factor *
        state<T> {
            std::min<T>( std::min<T>( std::abs( a[0] ), std::abs( b[0] ) ), std::abs( c[0] ) ),
            std::min<T>( std::min<T>( std::abs( a[1] ), std::abs( b[1] ) ), std::abs( c[1] ) ),
            std::min<T>( std::min<T>( std::abs( a[2] ), std::abs( b[2] ) ), std::abs( c[2] ) )
        }
    };
    // clang-format on
    return slope;
}

template <typename T, std::size_t Size>
constexpr T
pressure( const state<T, Size> & q, const T gamma ) {
    return ( gamma - 1 ) * ( q[2] - ( ( q[1] * q[1] ) / ( 2 * q[0] ) ) );
}

template <typename T, std::size_t Size>
constexpr std::array<T, Size>
pressure( const std::array<T, Size> & q1, const std::array<T, Size> & q2,
          const std::array<T, Size> & q3, const T gamma ) {
    return ( gamma - 1 ) * ( q3 - ( ( q2 * q2 ) / ( 2 * q1 ) ) );
}

template <typename T, std::size_t Size>
constexpr flux<T, Size>
f( const state<T, Size> & q, const T gamma ) {
    const auto p = pressure( q, gamma );
    // clang-format off
    const flux<T> f{
        q[1],
        ( q[1] * q[1] / q[0] ) + p,
        ( q[1] / q[0] ) * ( q[2] + p )
    };
    // clang-format on
    return f;
}

template <typename T>
constexpr inline T
v( const state<T> & q ) {
    return q[1] / q[0];
}

template <typename T, std::size_t Size>
constexpr std::array<T, Size>
v( const std::array<T, Size> & q1, const std::array<T, Size> & q2 ) {
    return q2 / q1;
}

template <typename T>
constexpr inline T
sound_speed( const state<T> & q, const T gamma ) {
    return std::sqrt( gamma * pressure( q, gamma ) / q[0] );
}

template <typename T>
constexpr inline T
max_wave_speed( const state<T> & q, const T gamma ) {
    return sound_speed( q, gamma ) + std::abs( q[1] / q[0] );
}

template <typename T>
constexpr inline T
e( const state<T> & q, const T gamma ) {
    return pressure( q, gamma ) / ( q[0] * ( gamma - 1 ) );
}

template <typename T, std::size_t Size>
constexpr std::array<T, Size>
e( const std::array<T, Size> & q1, const std::array<T, Size> & q2,
   const std::array<T, Size> & q3, const T gamma ) {
    return pressure( q1, q2, q3, gamma ) / ( q1 * ( gamma - 1 ) );
}

template <typename T, std::size_t Size>
constexpr std::array<state<T>, Size>
construct_state( const std::array<T, Size> & q1, const std::array<T, Size> & q2,
                 const std::array<T, Size> & q3 ) {
    std::array<state<T>, Size> state_array{};

    for ( std::size_t i{ 0 }; i < Size; ++i ) {
        state_array[i] = state<T>{ q1[i], q2[i], q3[i] };
    }

    return state_array;
}


template <typename T, std::size_t Size,
          approx_order Order = approx_order::first,
          const bool   minus_half = true>
constexpr inline state<T>
get_state( const std::array<state<T>, Size> & states, const std::size_t i,
           const T dx, const T gamma ) {
    // std::cout << "get_state: " << i - 1 << " " << i << " " << i + 1
    //           << std::endl;
    if ( Order == approx_order::second ) {
        if constexpr ( minus_half ) {
            return states[i]
                   - 0.5 * dx
                         * slope( states[i - 1], states[i], states[i + 1], dx,
                                  gamma );
        }
        else {
            return states[i]
                   + 0.5 * dx
                         * slope( states[i - 1], states[i], states[i + 1], dx,
                                  gamma );
        }
    }
    else {
        return states[i];
    }
}

template <typename T, std::size_t Size>
using fluid_algorithm =
    std::function<flux<T>( const std::array<T, Size> &, const std::size_t,
                           const T, const T, const T )>;

template <typename T, std::size_t Size>
constexpr flux<T>
lax_friedrichs( const std::array<state<T>, Size> & state, const std::size_t i,
                const T dt, const T dx, const T gamma ) {
    const auto f_i{ f( state[i], gamma ) }, f_i_1{ f( state[i + 1], gamma ) };
    // clang-format off
                const flux<T> f{
                    0.5 * (f_i + f_i_1 )
                    + 0.5 * ( dx / dt ) * ( state[i] - state[i + 1] )
                };
    // clang-format on
    return f;
}

template <typename T, std::size_t Size>
constexpr flux<T>
lax_wendroff( const std::array<state<T>, Size> & state, const std::size_t i,
              const T dt, const T dx, const T gamma ) {
    const auto & q{ state[i] };
    const auto & q_1{ state[i + 1] };
    return f( 0.5 * ( q + q_1 )
                  + 0.5 * ( dt / dx ) * ( f( q, gamma ) - f( q_1, gamma ) ),
              gamma );
}

template <typename T, std::size_t Size,
          approx_order Order = approx_order::first, bool minus_half = true>
constexpr flux<T>
hll( const std::array<state<T>, Size> & states, const std::size_t i,
     [[maybe_unused]] const T dt, [[maybe_unused]] const T dx, const T gamma ) {
    // std::cout << "hll: " << i << std::endl;
    //  L & R states
    state<T> U_L, U_R;
    if constexpr ( Order == approx_order::first ) {
        U_L = states[i - 1];
        U_R = states[i];
    }
    else {
        if constexpr ( minus_half ) {
            // std::cout << "minus_half" << std::endl;
            U_L = get_state<T, Size, Order, false>( states, i - 1, dx, gamma );
            U_R = get_state<T, Size, Order, true>( states, i, dx, gamma );
        }
        else {
            // std::cout << "plus_half" << std::endl;
            U_L = get_state<T, Size, Order, false>( states, i, dx, gamma );
            U_R = get_state<T, Size, Order, true>( states, i + 1, dx, gamma );
        }
    }

    // std::cout << "U: " << array_string( U_L ) << " " << array_string( U_R )
    //<< std::endl;

    // L & R velocities
    const auto v_L{ v( U_L ) }, v_R{ v( U_R ) };
    // std::cout << "v " << v_L << " " << v_R << std::endl;

    // L & R sound speed
    const auto c_L{ sound_speed( U_L, gamma ) },
        c_R{ sound_speed( U_R, gamma ) };
    // std::cout << "c: " << c_L << " " << c_R << std::endl;
    //  L, R, & * pressure
    const auto p_L{ pressure( U_L, gamma ) }, p_R{ pressure( U_R, gamma ) };
    const auto p_star{ 0.5 * ( p_L + p_R )
                       - 0.125 * ( v_R - v_L ) * ( U_R[0] - U_L[0] )
                             * ( c_R - c_L ) };
    // std::cout << "p: " << p_L << " " << p_star << " " << p_R << std::endl;
    //  L & R q
    const auto q_L{ p_star <= p_L ?
                        1 :
                        std::sqrt( 1
                                   + ( gamma + 1 ) * ( ( p_star / p_L ) - 1 )
                                         / ( 2 * gamma ) ) };
    const auto q_R{ p_star <= p_R ?
                        1 :
                        std::sqrt( 1
                                   + ( gamma + 1 ) * ( ( p_star / p_R ) - 1 )
                                         / ( 2 * gamma ) ) };
    // std::cout << "q " << q_L << " " << q_R << std::endl;
    //  L & R wavespeeds
    const auto S_L{ v_L - c_L * q_L }, S_R{ v_R + c_R * q_R };
    // std::cout << "S: " << S_L << " " << S_R << std::endl;

    const auto F_L{ f( U_L, gamma ) }, F_R{ f( U_R, gamma ) };
    // L, R & HLL fluxes
    if ( S_L > 0 ) {
        return F_L;
    }
    else if ( S_R > 0 && S_L < 0 ) {
        return ( S_R * F_L - S_L * F_R + S_L * S_R * ( U_R - U_L ) )
               / ( S_R - S_L );
    }
    else if ( S_R < 0 ) {
        return F_R;
    }
    else {
        std::cout << "ERROR: Invalid branch reached." << std::endl;
        // assert( false );
        return flux<T>{ 0., 0., 0. };
    }
}

template <typename T, std::size_t Size,
          approx_order Order = approx_order::first, bool minus_half = true>
constexpr flux<T>
hllc( const std::array<state<T>, Size> & states, const std::size_t i,
      [[maybe_unused]] const T dt, [[maybe_unused]] const T dx,
      const T gamma ) {
    // L & R states
    state<T> Q_L{}, Q_R{};
    if constexpr ( Order == approx_order::first ) {
        Q_L = states[i - 1];
        Q_R = states[i];
    }
    else {
        if constexpr ( minus_half ) {
            Q_L = get_state<T, Size, Order, false>( states, i - 1, dx, gamma );
            Q_R = get_state<T, Size, Order, true>( states, i, dx, gamma );
        }
        else {
            Q_L = get_state<T, Size, Order, false>( states, i, dx, gamma );
            Q_R = get_state<T, Size, Order, true>( states, i + 1, dx, gamma );
        }
    }
    // L & R velocities
    const auto v_L{ v( Q_L ) }, v_R{ v( Q_R ) };
    // L & R sound speed
    const auto c_L{ sound_speed( Q_L, gamma ) },
        c_R{ sound_speed( Q_R, gamma ) };
    // L, R, & * pressure
    const auto p_L{ pressure( Q_L, gamma ) }, p_R{ pressure( Q_R, gamma ) };
    // clang-format off
                // exponent z
                const T z = ( gamma - 1 ) / ( 2 * gamma );
                // p_star according to paper
                const auto p_star{
                    std::pow(
                        (c_L + c_R - 0.5 * (gamma - 1) * (v_R - v_L))
                        /
                        ((c_L / std::pow(pressure(Q_L, gamma), z)) + (c_R / std::pow(pressure(Q_R, gamma), z))),
                        1 / z
                    )
                };
    // clang-format on
    // L & R q
    const auto q_L{ p_star <= p_L ?
                        1 :
                        std::sqrt( 1
                                   + ( gamma + 1 ) * ( ( p_star / p_L ) - 1 )
                                         / ( 2 * gamma ) ) };
    const auto q_R{ p_star <= p_R ?
                        1 :
                        std::sqrt( 1
                                   + ( gamma + 1 ) * ( ( p_star / p_R ) - 1 )
                                         / ( 2 * gamma ) ) };
    // L & R wavespeeds
    const auto S_L{ v_L - c_L * q_L }, S_R{ v_R + c_R * q_R };
    // clang-format off
                const auto S_star{
                    ( p_R - p_L + Q_L[1] * ( S_L - v_L ) - Q_R[1] * ( S_R - v_R ) )
                    /
                    ( Q_L[0] * ( S_L - v_L ) - Q_R[0] * ( S_R - v_R ))
                };
    // clang-format on

    // L* & R* Q
    const auto Q_L_prefactor{ Q_L[0] * ( S_L - v_L ) / ( S_L - S_star ) },
        Q_R_prefactor{ Q_R[0] * ( S_R - v_R ) / ( S_R - S_star ) };
    // clang-format off
                const auto Q_L_star{
                    Q_L_prefactor
                    * state<T>{
                        1,
                        S_star,
                        ( Q_L[2] / Q_L[0] )
                          + ( S_star - v_L )
                              * ( S_star
                                  + ( p_L ) / ( Q_L[0] * ( S_L - v_L ) ) ) }
                };
                const auto Q_R_star{
                    Q_R_prefactor
                    * state<T>{ 1, S_star,
                                ( Q_R[2] / Q_R[0] )
                                    + ( S_star - v_R )
                                          * ( S_star
                                              + ( p_R )
                                                    / ( Q_R[0]
                                                        * ( S_R - v_R ) ) ) }
                };
    // clang-format on

    flux<T> F{};
    if ( 0 <= S_L ) {
        F = f( Q_L, gamma );
    }
    else if ( S_L <= 0 && 0 <= S_star ) {
        F = f( Q_L, gamma ) + S_L * ( Q_L_star - Q_L );
    }
    else if ( S_star <= 0 && 0 <= S_R ) {
        F = f( Q_R, gamma ) + S_R * ( Q_R_star - Q_R );
    }
    else if ( S_R <= 0 ) {
        F = f( Q_R, gamma );
    }

    return F;
}


#define ARRAY_SIZE( Size, Order ) \
    ( Size ) + 2 * ( static_cast<std::size_t>( ( Order ) ) )


// Fluid dynamics solver definition
template <typename T, std::size_t Size, solution_type Type, boundary_type Lbc,
          boundary_type Rbc, approx_order Order = approx_order::first,
          coordinate_type Coords = coordinate_type::cartesian,
          bool            incl_endpoint = true>
class fluid_solver
{
    public:
    fluid_solver( const T x_min, const T x_max,
                  const std::array<state<T>, Size> & initial_state ) :
        m_dx( ( x_max - x_min ) / ( incl_endpoint ? Size - 1 : Size ) ),
        m_offset( static_cast<std::size_t>( Order ) ),
        m_x( linspace<T, Size, incl_endpoint>( x_min, x_max ) ) {
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + m_offset] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    fluid_solver( const T x_min, const T x_max, const std::array<T, Size> & q1,
                  const std::array<T, Size> & q2,
                  const std::array<T, Size> & q3 ) :
        m_dx( ( x_max - x_min ) / ( incl_endpoint ? Size - 1 : Size ) ),
        m_offset( static_cast<std::size_t>( Order ) ),
        m_x( linspace<T, Size, incl_endpoint>( x_min, x_max ) ) {
        const auto initial_state{ construct_state( q1, q2, q3 ) };
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + m_offset] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }

    void initialize_state(
        const T x_min, const T x_max,
        const std::array<state<T>, Size> & initial_state ) noexcept {
        m_dx = ( x_max - x_min ) / ( incl_endpoint ? Size - 1 : Size );
        m_x = linspace<T, Size, incl_endpoint>( x_min, x_max );

        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + m_offset] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    void intialize_state( const T x_min, const T x_max,
                          const std::array<T, Size> & q1,
                          const std::array<T, Size> & q2,
                          const std::array<T, Size> & q3 ) {
        const auto initial_state{ construct_state( q1, q2, q3 ) };
        initialize_state( x_min, x_max, initial_state );
    }

    [[nodiscard]] constexpr auto & current_state() const noexcept {
        return m_state;
    }
    [[nodiscard]] constexpr auto & previous_state() const noexcept {
        return m_previous_state;
    }
    [[nodiscard]] constexpr auto q1() const noexcept {
        std::array<T, Size> q1_array{};
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            q1_array[i] = m_state[i + 1][0];
        }
        return q1_array;
    }
    [[nodiscard]] constexpr auto q2() const noexcept {
        std::array<T, Size> q2_array{};
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            q2_array[i] = m_state[i + 1][1];
        }
        return q2_array;
    }
    [[nodiscard]] constexpr auto q3() const noexcept {
        std::array<T, Size> q3_array{};
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            q3_array[i] = m_state[i + 1][2];
        }
        return q3_array;
    }
    [[nodiscard]] constexpr auto dx() const noexcept { return m_dx; }
    [[nodiscard]] constexpr auto x() const noexcept { return m_x; }

    constexpr auto simulate( const T endpoint, const T gamma,
                             const bool  save_endpoint = true,
                             std::string opt_id = "" ) noexcept;

    private:
    [[nodiscard]] constexpr auto apply_boundary_conditions() noexcept;
    [[nodiscard]] constexpr std::array<state<T>, ARRAY_SIZE( Size, Order )>
    d_state( const std::array<state<T>, ARRAY_SIZE( Size, Order )> & states,
             [[maybe_unused]] const T t, const T dt, const T gamma ) noexcept;

    T           m_dx;     // Difference between cells
    std::size_t m_offset; // Offset due to ghost cells
    // An array of state arrays (3 values) with length Size + 2, 1 ghost cell
    // each side
    std::array<state<T>, ARRAY_SIZE( Size, Order )> m_state;
    std::array<state<T>, ARRAY_SIZE( Size, Order )> m_previous_state;
    std::array<T, Size>                             m_x;
};

template <typename T, std::size_t Size, solution_type Type, boundary_type Lbc,
          boundary_type Rbc, approx_order Order, coordinate_type Coords,
          bool incl_endpoint>
constexpr auto
fluid_solver<T, Size, Type, Lbc, Rbc, Order, Coords, incl_endpoint>::simulate(
    const T endpoint, const T gamma, const bool save_endpoint,
    std::string opt_id ) noexcept {
    if ( !opt_id.empty() ) {
        opt_id += "_";
    }

    const auto CFL_condition = [*this, &gamma]() {
        std::array<T, Size> s_max;
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            s_max[i] = max_wave_speed( m_state[i + 1], gamma );
        }
        const auto max = *std::max_element( s_max.cbegin(), s_max.cend() );
        return 0.3 * m_dx / max;
    };

    T time_step = CFL_condition();
    for ( T t{ 0 }; t <= endpoint; t += time_step ) {
        const auto K1 = time_step * d_state( m_state, t, time_step, gamma );
        const auto K2 =
            time_step
            * d_state( m_state + K1, t + time_step, time_step, gamma );
        m_state = m_state + 0.5 * ( K1 + K2 );

        if constexpr ( Coords == coordinate_type::spherical ) {
            for ( std::size_t i{ m_offset }; i < Size + m_offset; ++i ) {
                auto &        Q{ m_state[i] };
                const flux<T> spherical_source{
                    2 * Q[1] / ( m_x[i - 1] + 0.25 * m_dx ),
                    2 * Q[0] * v( Q ) * v( Q ) / ( m_x[i - 1] + 0.25 * m_dx ),
                    2 * ( Q[2] + pressure( Q, gamma ) ) * v( Q )
                        / ( m_x[i - 1] + 0.25 * m_dx )
                };
                Q -= spherical_source * time_step;
            }
        }

        m_previous_state = m_state;
        apply_boundary_conditions();
        time_step = CFL_condition();
    }

    if ( save_endpoint ) {
        const auto Q1{ q1() };
        const auto Q2{ q2() };
        const auto Q3{ q3() };
        write_to_file<T, Size>(
            opt_id + std::to_string( endpoint ) + "s_"
                + solution_string[static_cast<std::size_t>( Type )] + "_"
                + approx_string[static_cast<std::size_t>( Order ) - 1]
                + "_state.csv",
            { m_x, Q1, v( Q1, Q2 ), pressure( Q1, Q2, Q3, gamma ),
              e( Q1, Q2, Q3, gamma ) },
            { "x", "d", "v", "p", "e" } );
    }

    return m_state;
}

template <typename T, std::size_t Size, solution_type Type, boundary_type Lbc,
          boundary_type Rbc, approx_order Order, coordinate_type Coords,
          bool incl_endpoint>
[[nodiscard]] constexpr auto
fluid_solver<T, Size, Type, Lbc, Rbc, Order, Coords,
             incl_endpoint>::apply_boundary_conditions() noexcept {
    switch ( Lbc ) {
    case boundary_type::outflow: {
        std::for_each_n( m_state.begin(), m_offset,
                         [*this]( auto & x ) { x = m_state[m_offset]; } );
        std::for_each_n(
            m_previous_state.begin(), m_offset,
            [*this]( auto & x ) { x = m_previous_state[m_offset]; } );
    } break;
    case boundary_type::reflecting: {
        std::for_each_n( m_state.begin(), m_offset, [*this]( auto & x ) {
            x = m_state[m_offset];
            x[1] *= -1;
        } );
        std::for_each_n( m_previous_state.begin(), m_offset,
                         [*this]( auto & x ) {
                             x = m_previous_state[m_offset];
                             x[1] *= -1;
                         } );
    } break;
    }

    switch ( Rbc ) {
    case boundary_type::outflow: {
        std::for_each_n(
            m_state.begin() + m_offset + Size, m_offset,
            [*this]( auto & x ) { x = m_state[m_offset + Size - 1]; } );
        std::for_each_n( m_previous_state.begin() + m_offset + Size, m_offset,
                         [*this]( auto & x ) {
                             x = m_previous_state[m_offset + Size - 1];
                         } );
    } break;
    case boundary_type::reflecting: {
        std::for_each_n( m_state.begin() + m_offset + Size, m_offset,
                         [*this]( auto & x ) {
                             x = m_state[m_offset];
                             x[1] *= -1.;
                         } );
        std::for_each_n( m_previous_state.begin() + m_offset + Size, m_offset,
                         [*this]( auto & x ) {
                             x = m_previous_state[m_offset];
                             x[1] *= -1.;
                         } );
    } break;
    }
}

template <typename T, std::size_t Size, solution_type Type, boundary_type Lbc,
          boundary_type Rbc, approx_order Order, coordinate_type Coords,
          bool incl_endpoint>
[[nodiscard]] constexpr std::array<state<T>, ARRAY_SIZE( Size, Order )>
fluid_solver<T, Size, Type, Lbc, Rbc, Order, Coords, incl_endpoint>::d_state(
    const std::array<state<T>, ARRAY_SIZE( Size, Order )> & states,
    [[maybe_unused]] const T t, const T dt, const T gamma ) noexcept {
    fluid_algorithm<T, ARRAY_SIZE( Size, Order )>   f_half;
    std::array<state<T>, ARRAY_SIZE( Size, Order )> delta{};
    for ( std::size_t i{ m_offset }; i < Size + m_offset; ++i ) {
        if constexpr ( Type == solution_type::lax_friedrichs ) {
            delta[i] = -( 1 / m_dx )
                       * ( lax_friedrichs( states, i, dt, m_dx, gamma )
                           - lax_friedrichs( states, i - 1, dt, m_dx, gamma ) );
        }
        else if constexpr ( Type == solution_type::lax_wendroff ) {
            delta[i] = -( 1 / m_dx )
                       * ( lax_wendroff( states, i, dt, m_dx, gamma )
                           - lax_wendroff( states, i - 1, dt, m_dx, gamma ) );
        }
        else if constexpr ( Type == solution_type::hll ) {
            delta[i] = -( 1 / m_dx )
                       * ( hll<T, ARRAY_SIZE( Size, Order ), Order, false>(
                               states, i + 1, dt, m_dx, gamma )
                           - hll<T, ARRAY_SIZE( Size, Order ), Order, true>(
                               states, i, dt, m_dx, gamma ) );
        }
        else if constexpr ( Type == solution_type::hllc ) {
            delta[i] = -( 1 / m_dx )
                       * ( hllc<T, ARRAY_SIZE( Size, Order ), Order, false>(
                               states, i + 1, dt, m_dx, gamma )
                           - hllc<T, ARRAY_SIZE( Size, Order ), Order, true>(
                               states, i, dt, m_dx, gamma ) );
        }
    }

    return delta;
}

int
main() {
    std::array<double, 100>                               q1;
    std::array<double, std::tuple_size_v<decltype( q1 )>> q2;
    std::array<double, std::tuple_size_v<decltype( q1 )>> q3;

    const double xmin{ 0. };
    const double xmax{ 1. };
    const double dx{ ( xmax - xmin ) / std::tuple_size_v<decltype( q1 )> };

    const double gamma{ 1.4 };
    double       rho{ 0. };
    double       p{ 0. };
    double       v{ 0. };

    // Set-up shocktube A:
    for ( std::size_t i{ 0 }; i < q1.size(); ++i ) {
        const double x = xmin + i * dx;
        if ( x < 0.3 ) {
            rho = 1.;
            v = 0.75;
            p = 1.;
        }
        else {
            rho = 0.125;
            v = 0.;
            p = 0.1;
        }

        const double epsilon = p / ( rho * ( gamma - 1 ) );

        q1[i] = rho;
        q2[i] = rho * v;
        q3[i] = rho * epsilon + 0.5 * rho * v * v;
    }

    auto initial_state = construct_state( q1, q2, q3 );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_friedrichs, boundary_type::outflow,
                 boundary_type::outflow, approx_order::second>
        fs_A_lf( xmin, xmax, initial_state );
    fs_A_lf.simulate( 0.2, gamma, true, "A" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_wendroff, boundary_type::outflow,
                 boundary_type::outflow, approx_order::second>
        fs_A_lw( xmin, xmax, initial_state );
    fs_A_lw.simulate( 0.2, gamma, true, "A" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>, solution_type::hll,
                 boundary_type::outflow, boundary_type::outflow,
                 approx_order::first>
        fs_A_hll( xmin, xmax, initial_state );
    fs_A_hll.simulate( 0.2, gamma, true, "A" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>, solution_type::hllc,
                 boundary_type::outflow, boundary_type::outflow,
                 approx_order::first>
        fs_A_hllc( xmin, xmax, initial_state );
    fs_A_hllc.simulate( 0.2, gamma, true, "A" );

    // Set-up shocktube B:
    for ( std::size_t i{ 0 }; i < q1.size(); ++i ) {
        const double x = xmin + i * dx;
        if ( x < 0.8 ) {
            rho = 1.;
            v = -19.59745;
            p = 1000.;
        }
        else {
            rho = 1.;
            v = -19.59745;
            p = 0.01;
        }

        const double epsilon = p / ( rho * ( gamma - 1 ) );

        q1[i] = rho;
        q2[i] = rho * v;
        q3[i] = rho * epsilon + 0.5 * rho * v * v;
    }

    initial_state = construct_state( q1, q2, q3 );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_friedrichs, boundary_type::outflow,
                 boundary_type::outflow, approx_order::second>
        fs_B_lf( xmin, xmax, initial_state );
    fs_B_lf.simulate( 0.012, gamma, true, "B" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_wendroff, boundary_type::outflow,
                 boundary_type::outflow, approx_order::second>
        fs_B_lw( xmin, xmax, initial_state );
    fs_B_lw.simulate( 0.012, gamma, true, "B" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>, solution_type::hll,
                 boundary_type::outflow, boundary_type::outflow,
                 approx_order::first>
        fs_B_hll( xmin, xmax, initial_state );
    fs_B_hll.simulate( 0.012, gamma, true, "B" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>, solution_type::hllc,
                 boundary_type::outflow, boundary_type::outflow,
                 approx_order::first>
        fs_B_hllc( xmin, xmax, initial_state );
    fs_B_hllc.simulate( 0.012, gamma, true, "B" );

    // Set-up spherical shocktube:
    for ( std::size_t i{ 0 }; i < q1.size(); ++i ) {
        const double r{ xmin + i * dx };

        if ( r < 0.4 ) {
            rho = 1.;
            p = 1.;
            v = 0.;
        }
        else {
            rho = 0.125;
            p = 0.1;
            v = 0.;
        }

        const double epsilon = p / ( rho * ( gamma - 1 ) );

        q1[i] = rho;
        q2[i] = rho * v;
        q3[i] = rho * epsilon + 0.5 * rho * v * v;
    }

    initial_state = construct_state( q1, q2, q3 );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_friedrichs, boundary_type::outflow,
                 boundary_type::outflow, approx_order::second,
                 coordinate_type::spherical>
        fs_spherical_lf( xmin, xmax, initial_state );
    fs_spherical_lf.simulate( 0.25, gamma, true, "S" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_wendroff, boundary_type::outflow,
                 boundary_type::outflow, approx_order::second,
                 coordinate_type::spherical>
        fs_spherical_lw( xmin, xmax, initial_state );
    fs_spherical_lw.simulate( 0.25, gamma, true, "S" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>, solution_type::hll,
                 boundary_type::outflow, boundary_type::outflow,
                 approx_order::first, coordinate_type::spherical>
        fs_spherical_hll( xmin, xmax, initial_state );
    fs_spherical_hll.simulate( 0.25, gamma, true, "S" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>, solution_type::hllc,
                 boundary_type::outflow, boundary_type::outflow,
                 approx_order::first, coordinate_type::spherical>
        fs_spherical_hllc( xmin, xmax, initial_state );
    fs_spherical_hllc.simulate( 0.25, gamma, true, "S" );
}
