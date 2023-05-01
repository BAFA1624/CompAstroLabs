#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Stringify things :)
#define STR( s ) #s

// Define some helper functions so vectors can be used to do memberwise:
//  - Addition
//  - Subtraction
//  - Multiplication
//  - Division

#define REF_OP_ARR_ARR( op )                                                  \
    template <typename T>                                                     \
    inline constexpr std::vector<T> & operator op##=(                         \
        std::vector<T> & l, const std::vector<T> & r ) {                      \
        assert( l.size() <= r.size() );                                       \
        for ( std::uint64_t i{ 0 }; i < l.size(); ++i ) { l[i] op## = r[i]; } \
        return l;                                                             \
    }
#define REF_OP_ARR_CONST( op )                                              \
    template <typename T1, typename T2>                                     \
    inline constexpr std::vector<T1> & operator op##=( std::vector<T1> & l, \
                                                       const T2          r ) {       \
        for ( std::uint64_t i{ 0 }; i < l.size(); ++i ) {                   \
            l[i] op## = static_cast<T1>( r );                               \
        }                                                                   \
        return l;                                                           \
    }
#define VAL_OP_ARR_ARR( op )                                                  \
    template <typename T>                                                     \
    inline constexpr std::vector<T> operator op( const std::vector<T> & l,    \
                                                 const std::vector<T> & r ) { \
        std::vector<T> v{ l };                                                \
        return ( v op## = r );                                                \
    }
#define VAL_OP_ARR_CONST( op )                                               \
    template <typename T1, typename T2>                                      \
    inline constexpr std::vector<T1> operator op( const std::vector<T1> & l, \
                                                  const T2                r ) {             \
        std::vector<T1> v{ l };                                              \
        for ( std::uint64_t i{ 0 }; i < l.size(); ++i ) {                    \
            v[i] op## = static_cast<T1>( r );                                \
        }                                                                    \
        return v;                                                            \
    }
#define VAL_OP_CONST_ARR( op )                            \
    template <typename T1, typename T2>                   \
    inline constexpr std::vector<T2> operator op(         \
        const T1 l, const std::vector<T2> & r ) {         \
        std::vector<T2> v{ r };                           \
        for ( std::uint64_t i{ 0 }; i < r.size(); ++i ) { \
            v[i] op## = static_cast<T2>( l );             \
        }                                                 \
        return v;                                         \
    }
#define DEF_VECTOR_OP( op ) \
    REF_OP_ARR_ARR( op )    \
    REF_OP_ARR_CONST( op )  \
    VAL_OP_ARR_ARR( op )    \
    VAL_OP_ARR_CONST( op )  \
    VAL_OP_CONST_ARR( op )

// clang-format off
DEF_VECTOR_OP( + )
DEF_VECTOR_OP( - )
DEF_VECTOR_OP( * )
DEF_VECTOR_OP( / )
// clang-format on


template <typename T, bool endpoint = true>
constexpr std::vector<T>
linspace( const T min, const T max, const std::uint64_t N ) {
    const T dx{ ( max - min ) / static_cast<T>( endpoint ? N - 1 : N ) };

    std::vector<T> vec( N );
    for ( std::uint64_t i{ 0 }; i < N; ++i ) { vec[i] = min + i * dx; }

    return vec;
}

// Simple std::array printer
template <typename T>
std::string
vector_string( const std::vector<T> & a ) {
    std::string s{ "vector: { " };
    for ( std::uint64_t i{ 0 }; i < a.size(); ++i ) {
        s += std::to_string( a[i] ) + ", ";
    }
    s.pop_back();
    s.pop_back();
    s += " }";
    return s;
}

// Writes a set of arrays to file
template <typename T>
void
write_to_file( const std::string &                 filename,
               const std::vector<std::vector<T>> & values,
               const std::vector<std::string> &    keys = {} ) {
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
template <typename T>
using state = std::vector<T>;
template <typename T>
using flux = state<T>;

template <typename T>
constexpr inline T
pressure( const state<T> & q, const T gamma ) {
    return ( gamma - 1 ) * ( q[2] - ( ( q[1] * q[1] ) / ( 2 * q[0] ) ) );
}

template <typename T>
constexpr inline flux<T>
f( const state<T> & q, const T gamma ) {
    const auto p = pressure( q, gamma );
    // clang-format off
    const flux<T> f{
        q[1],
        ( q[1] * q[1] / q[0] ) + p,
        (q[1] / q[0]) * (q[2] + p)
    };
    // clang-format on
    return f;
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

template <typename T>
constexpr std::vector<state<T>>
construct_state( const std::vector<T> & q1, const std::vector<T> & q2,
                 const std::vector<T> & q3 ) {
    assert( q1.size() == q2.size() );
    assert( q2.size() == q3.size() );

    std::vector<state<T>> state_array( q1.size() );

    for ( std::size_t i{ 0 }; i < q1.size(); ++i ) {
        state_array[i] = state<T>{ q1[i], q2[i], q3[i] };
    }

    return state_array;
}

// Class enum for selecting between type of algorithm.
// The value of each enum name is set to the required no. of ghost cells
enum class solution_type : std::uint64_t { lax_friedrichs, lax_wendroff, hll };
enum class boundary_type : std::uint64_t { outflow, reflecting, custom };
const std::vector<std::string> solution_string{ "lax_friedrichs",
                                                "lax_wendroff", "hll" };

// Fluid dynamics solver definition
template <typename T, solution_type Type, boundary_type Lbc, boundary_type Rbc,
          bool incl_endpoint = true>
class fluid_solver
{
    public:
    fluid_solver( const T x_min, const T x_max,
                  const std::vector<state<T>> & initial_state ) :
        m_dx( ( x_max - x_min )
              / ( incl_endpoint ? initial_state.size() - 1 :
                                  initial_state.size() ) ),
        m_x( linspace<T, incl_endpoint>( x_min, x_max, initial_state.size() ) ),
        m_state( initial_state.size() + 2 ),
        m_previous_state( initial_state.size() + 2 ) {
        for ( std::uint64_t i{ 0 }; i < initial_state.size(); ++i ) {
            m_state[i + 1] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    fluid_solver( const T x_min, const T x_max, const std::vector<T> & q1,
                  const std::vector<T> & q2, const std::vector<T> & q3 ) :
        m_dx( ( x_max - x_min )
              / ( incl_endpoint ? q1.size() - 1 : q1.size() ) ),
        m_x( linspace<T, incl_endpoint>( x_min, x_max, q1.size() ) ),
        m_state( q1.size() + 2 ),
        m_previous_state( q1.size() + 2 ) {
        const auto initial_state{ construct_state( q1, q2, q3 ) };
        for ( std::size_t i{ 0 }; i < initial_state.size(); ++i ) {
            m_state[i + 1] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }

    void
    initialize_state( const T x_min, const T x_max,
                      const std::vector<state<T>> & initial_state ) noexcept {
        m_dx = ( x_max - x_min )
               / ( incl_endpoint ? initial_state.size() - 1 :
                                   initial_state.size() );
        m_x = linspace<T, incl_endpoint>( x_min, x_max );

        std::vector<T> tmp( initial_state.size() + 2 );
        for ( std::uint64_t i{ 0 }; i < initial_state.size(); ++i ) {
            tmp[i + 1] = initial_state[i];
        }
        m_state = std::move( tmp );
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    void intialize_state( const T x_min, const T x_max,
                          const std::vector<T> & q1, const std::vector<T> & q2,
                          const std::vector<T> & q3 ) {
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
        std::vector<T> q1_array( m_x.size() );
        for ( std::uint64_t i{ 0 }; i < m_x.size(); ++i ) {
            q1_array[i] = m_state[i + 1][0];
        }
        return q1_array;
    }
    [[nodiscard]] constexpr auto q2() const noexcept {
        std::vector<T> q2_array( m_x.size() );
        for ( std::uint64_t i{ 0 }; i < m_x.size(); ++i ) {
            q2_array[i] = m_state[i + 1][1];
        }
        return q2_array;
    }
    [[nodiscard]] constexpr auto q3() const noexcept {
        std::vector<T> q3_array( m_x.size() );
        for ( std::uint64_t i{ 0 }; i < m_x.size(); ++i ) {
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
            std::vector<T> s_max( m_x.size() );
            for ( std::uint64_t i{ 0 }; i < m_x.size(); ++i ) {
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
            write_to_file<T>(
                opt_id + std::to_string( endpoint ) + "s_"
                    + solution_string[static_cast<std::uint64_t>( Type )]
                    + "_state.csv",
                { m_x, q1(), q2(), q3() }, { "x", "q1", "q2", "q3" } );
        }

        return m_state;
    }

    private:
    constexpr void apply_boundary_conditions() noexcept {
        const std::uint64_t size{ m_x.size() };
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
            m_state[size + 1] = m_state[size];
            m_previous_state[size + 1] = m_previous_state[size];
        } break;
        case boundary_type::reflecting: {
            m_state[size + 1] = m_state[size];
            m_previous_state[size + 1] = m_previous_state[size];
            m_state[size + 1][1] *= -1;
            m_previous_state[size + 1][1] *= -1;
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

            for ( std::size_t i{ 1 }; i <= m_x.size(); ++i ) {
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

            for ( std::size_t i{ 1 }; i <= m_x.size(); ++i ) {
                m_state[i] = m_previous_state[i]
                             - ( time_step / m_dx )
                                   * ( f( q_half( i ), gamma )
                                       - f( q_half( i - 1 ), gamma ) );
            }
        }
    }

    // x resolution
    T m_dx;
    // Associated x values each non-ghost cell.
    std::vector<T> m_x;
    // Vectors for the current & previous state
    std::vector<state<T>> m_state;
    std::vector<state<T>> m_previous_state;
};

int
main() {
    const double gamma = 1.4;

    const auto xarr{ linspace<double>( 0, 1, 100 ) };

    std::vector<double> q1( xarr.size() );
    std::vector<double> q2( xarr.size() );
    std::vector<double> q3( xarr.size() );
    for ( std::uint64_t i{ 0 }; i < xarr.size(); ++i ) {
        const auto & x = xarr[i];
        double       rho{ 0 }, v{ 0 }, epsilon{ 0 }, p{ 0 };
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

        epsilon = p / ( rho * ( gamma - 1 ) );

        q1[i] = rho;
        q2[i] = rho * v;
        q3[i] = rho * epsilon + 0.5 * rho * v * v;
    }

    const auto initial_state = construct_state( q1, q2, q3 );

    fluid_solver<double, solution_type::lax_friedrichs, boundary_type::outflow,
                 boundary_type::outflow>
               fs_lf( 0., 1., initial_state );
    const auto state_A_lf = fs_lf.simulate( 0.2, gamma, true, "A" );

    fluid_solver<double, solution_type::lax_wendroff, boundary_type::outflow,
                 boundary_type::outflow>
               fs_lw( 0., 1., initial_state );
    const auto state_A_lw = fs_lw.simulate( 0.2, gamma, true, "A" );
}