#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
#define REF_OP_ARR_CONST( op )                            \
    template <typename T1, typename T2, std::size_t Size> \
    inline constexpr std::array<T1, Size> & operator op(  \
        std::array<T1, Size> & lhs, const T2 rhs ) {      \
        for ( std::size_t i{ 0 }; i < Size; ++i ) {       \
            lhs[i] op static_cast<T1>( rhs );             \
        }                                                 \
        return lhs;                                       \
    }
#define REF_OP_ARR_CONST_3( op )                                               \
    template <typename T1, typename T2>                                        \
    inline constexpr std::array<T1, 3> & operator op( std::array<T1, 3> & lhs, \
                                                      const T2 rhs ) {         \
        lhs[0] op static_cast<T1>( rhs );                                      \
        lhs[1] op static_cast<T1>( rhs );                                      \
        lhs[2] op static_cast<T1>( rhs );                                      \
        return lhs;                                                            \
    }
#define REF_OP_CONST_ARR( op )                            \
    template <typename T1, typename T2, std::size_t Size> \
    inline constexpr std::array<T2, Size> & operator op(  \
        const T1 lhs, std::array<T2, Size> & rhs ) {      \
        for ( std::size_t i = 0; i < Size; ++i ) {        \
            rhs[i] op static_cast<T2>( lhs );             \
        }                                                 \
        return lhs;                                       \
    }
#define REF_OP_CONST_ARR_3( op )                      \
    template <typename T1, typename T2>               \
    inline constexpr std::array<T2, 3> & operator op( \
        const T1 lhs, std::array<T2, 3> & rhs ) {     \
        rhs[0] op static_cast<T2>( lhs );             \
        rhs[1] op static_cast<T2>( lhs );             \
        rhs[2] op static_cast<T2>( lhs );             \
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
        for ( std::size_t i{ 0 }; i < Size; ++i ) {                            \
            tmp[i] op## = static_cast<T1>( rhs );                              \
        }                                                                      \
        return tmp;                                                            \
    }
#define VAL_OP_ARR_CONST_3( op )                        \
    template <typename T1, typename T2>                 \
    inline constexpr std::array<T1, 3> operator op(     \
        const std::array<T1, 3> & lhs, const T2 rhs ) { \
        auto tmp{ lhs };                                \
        tmp[0] op## = static_cast<T1>( rhs );           \
        tmp[1] op## = static_cast<T1>( rhs );           \
        tmp[2] op## = static_cast<T1>( rhs );           \
        return tmp;                                     \
    }
#define VAL_OP_CONST_ARR( op )                             \
    template <typename T1, typename T2, std::size_t Size>  \
    inline constexpr std::array<T2, Size> operator op(     \
        const T1 lhs, const std::array<T2, Size> & rhs ) { \
        auto tmp{ rhs };                                   \
        for ( std::size_t i{ 0 }; i < Size; ++i ) {        \
            tmp[i] op## = static_cast<T2>( lhs );          \
        }                                                  \
        return tmp;                                        \
    }
#define VAL_OP_CONST_ARR_3( op )                        \
    template <typename T1, typename T2>                 \
    inline constexpr std::array<T2, 3> operator op(     \
        const T1 lhs, const std::array<T2, 3> & rhs ) { \
        auto tmp{ rhs };                                \
        tmp[0] op## = static_cast<T2>( lhs );           \
        tmp[1] op## = static_cast<T2>( lhs );           \
        tmp[2] op## = static_cast<T2>( lhs );           \
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

template <typename T, std::size_t Size>
constexpr std::array<T, Size>
v( const std::array<T, Size> & q1, const std::array<T, Size> & q2 ) {
    return q2 / q1;
}

template <typename T>
constexpr inline T
sound_speed( const state<T> & q, const T gamma ) {
    return std::sqrt( std::abs( gamma * pressure( q, gamma ) / q[0] ) );
}

template <typename T>
constexpr inline T
max_wave_speed( const state<T> & q, const T gamma ) {
    return sound_speed( q, gamma ) + std::abs( q[1] / q[0] );
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

// Class enum for selecting between type of algorithm.
// The value of each enum name is set to the required no. of ghost cells
enum class solution_type : std::size_t { lax_friedrichs, lax_wendroff, hll };
enum class boundary_type : std::size_t { outflow, reflecting, custom };

const std::array<std::string, 3> solution_string{ "lax_friedrichs",
                                                  "lax_wendroff", "hll" };


// Fluid dynamics solver definition
template <typename T, std::size_t Size, solution_type Type, boundary_type Lbc,
          boundary_type Rbc, bool incl_endpoint = true>
class fluid_solver
{
    public:
    fluid_solver( const T x_min, const T x_max,
                  const std::array<state<T>, Size> & initial_state ) :
        m_dx( ( x_max - x_min ) / ( incl_endpoint ? Size - 1 : Size ) ),
        m_x( linspace<T, Size, incl_endpoint>( x_min, x_max ) ) {
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + 1] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    fluid_solver( const T x_min, const T x_max, const std::array<T, Size> & q1,
                  const std::array<T, Size> & q2,
                  const std::array<T, Size> & q3 ) :
        m_dx( ( x_max - x_min ) / ( incl_endpoint ? Size - 1 : Size ) ),
        m_x( linspace<T, Size, incl_endpoint>( x_min, x_max ) ) {
        const auto initial_state{ construct_state( q1, q2, q3 ) };
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + 1] = initial_state[i];
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
            m_state[i + 1] = initial_state[i];
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
        std::array<T, Size> q1_array;
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            q1_array[i] = m_state[i + 1][0];
        }
        return q1_array;
    }
    [[nodiscard]] constexpr auto q2() const noexcept {
        std::array<T, Size> q2_array;
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            q2_array[i] = m_state[i + 1][1];
        }
        return q2_array;
    }
    [[nodiscard]] constexpr auto q3() const noexcept {
        std::array<T, Size> q3_array;
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            q3_array[i] = m_state[i + 1][2];
        }
        return q3_array;
    }
    [[nodiscard]] constexpr auto dx() const noexcept { return m_dx; }
    [[nodiscard]] constexpr auto x() const noexcept { return m_x; }

    constexpr auto simulate( const T endpoint, const T gamma,
                             const bool  save_endpoint = true,
                             std::string opt_id = "" ) noexcept {
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
            update_state( time_step, gamma );
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
                    + solution_string[static_cast<std::size_t>( Type )]
                    + "_state.csv",
                { m_x, Q1, v( Q1, Q2 ), pressure( Q1, Q2, Q3, gamma ),
                  e( Q1, Q2, Q3, gamma ) },
                { "x", "d", "v", "p", "e" } );
        }

        return m_state;
    }

    private:
    constexpr void apply_boundary_conditions() noexcept {
        switch ( Lbc ) {
        case boundary_type::outflow: {
            m_state[0] = m_state[1];
            m_previous_state[0] = m_previous_state[1];
        } break;
        case boundary_type::reflecting: {
            m_state[0] = m_state[1];
            m_previous_state[0] = m_previous_state[1];
            m_state[0][1] *= -1;
            m_previous_state[0][1] *= -1;
        } break;
        }
        switch ( Rbc ) {
        case boundary_type::outflow: {
            m_state[Size + 1] = m_state[Size];
            m_previous_state[Size + 1] = m_previous_state[Size];
        } break;
        case boundary_type::reflecting: {
            m_state[Size + 1] = m_state[Size];
            m_previous_state[Size + 1] = m_previous_state[Size];
            m_state[Size + 1][1] *= -1;
            m_previous_state[Size + 1][1] *= -1;
        } break;
        }
    };

    constexpr void update_state( const T time_step, const T gamma ) noexcept {
        if constexpr ( Type == solution_type::lax_friedrichs ) {
            const auto f_half = [*this, time_step,
                                 gamma]( const std::size_t i ) {
                const auto f_i{ f( m_previous_state[i], gamma ) },
                    f_i_1{ f( m_previous_state[i + 1], gamma ) };
                // clang-format off
                const flux<T> f{
                    0.5 * (f_i + f_i_1 )
                    + 0.5 * ( m_dx / time_step ) * ( m_previous_state[i] - m_previous_state[i + 1] )
                };
                // clang-format on
                return f;
            };

            for ( std::size_t i{ 1 }; i <= Size; ++i ) {
                m_state[i] =
                    m_previous_state[i]
                    + ( time_step / m_dx ) * ( f_half( i - 1 ) - f_half( i ) );
            }
        }
        else if constexpr ( Type == solution_type::lax_wendroff ) {
            const auto q_half = [*this, &gamma,
                                 &time_step]( const std::uint64_t i ) {
                const auto & q{ m_previous_state[i] };
                const auto & q_1{ m_previous_state[i + 1] };
                return 0.5 * ( q + q_1 )
                       + 0.5 * ( time_step / m_dx )
                             * ( f( q, gamma ) - f( q_1, gamma ) );
            };

            for ( std::size_t i{ 1 }; i <= Size; ++i ) {
                m_state[i] = m_previous_state[i]
                             - ( time_step / m_dx )
                                   * ( f( q_half( i ), gamma )
                                       - f( q_half( i - 1 ), gamma ) );
            }
        }
    }

    T m_dx;
    // An array of state arrays (3 values) with length Size + 2, 1 ghost cell
    // each side
    std::array<state<T>, Size + 2> m_state;
    std::array<state<T>, Size + 2> m_previous_state;
    std::array<T, Size>            m_x;
};

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
                 boundary_type::outflow>
        fs_A_lf( xmin, xmax, initial_state );
    fs_A_lf.simulate( 0.2, gamma, true, "A" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_wendroff, boundary_type::outflow,
                 boundary_type::outflow>
        fs_A_lw( xmin, xmax, initial_state );
    fs_A_lw.simulate( 0.2, gamma, true, "A" );

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
                 boundary_type::outflow>
        fs_B_lf( xmin, xmax, initial_state );
    fs_B_lf.simulate( 0.012, gamma, true, "B" );

    fluid_solver<double, std::tuple_size_v<decltype( q1 )>,
                 solution_type::lax_wendroff, boundary_type::outflow,
                 boundary_type::outflow>
        fs_B_lw( xmin, xmax, initial_state );
    fs_B_lw.simulate( 0.012, gamma, true, "B" );

    // Set-up spherical shocktube:
    // for ( std::size_t i{ 0 }; i < q1.size(); ++i ) {}
}
