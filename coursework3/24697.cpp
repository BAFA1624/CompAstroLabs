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
/*REF_OP( += )
REF_OP( -= )
REF_OP( *= )
REF_OP( /= )
VAL_OP( + )
VAL_OP( - )
VAL_OP( * )
VAL_OP( / )*/
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
get_pressure( const state<T, Size> & q, const T gamma ) {
    // (gamma - 1) * rho * epsilon
    return ( gamma - 1 ) * q[0]
           * ( ( q[2] / q[0] ) - ( q[1] * q[1] / ( 2 * q[0] * q[0] ) ) );
}

template <typename T, std::size_t Size>
constexpr flux<T, Size>
get_flux( const state<T, Size> & q, const T gamma ) {
    const auto p = get_pressure( q, gamma );
    return flux<T, Size>{ q[1], ( q[1] * q[1] / q[0] ) + p,
                          ( q[2] / q[0] ) * ( q[2] + p ) };
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
enum class solution_type : std::size_t { lax_friedrichs, law_wendroff };
enum class boundary_type : std::size_t { outflow, reflecting, custom };
std::array<std::string, 2> solution_string{ "lax_friedrichs", "lax_wendroff" };

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

    constexpr auto simulate( const T time_step, const T endpoint, const T gamma,
                             const std::size_t n_saves = 1,
                             const bool        save_endpoint = true,
                             std::string       opt_id = "" ) noexcept {
        auto save_count{ static_cast<std::size_t>( save_endpoint ? 1 : 0 ) };
        const auto n_steps{ static_cast<std::size_t>( endpoint / time_step ) };
        const auto save_state_freq{ static_cast<std::size_t>(
            n_steps / ( save_endpoint ? n_saves - 1 : n_saves ) ) };

        if ( !opt_id.empty() ) {
            opt_id += "_";
        }

        for ( T t{ 0 }; t <= endpoint; t += time_step ) {
            update_state( time_step, gamma );

            m_previous_state = m_state;

            apply_boundary_conditions();

            if ( save_count < n_saves
                 && static_cast<std::size_t>( t / time_step ) % save_state_freq
                        == 0 ) {
                write_to_file<double, Size>(
                    opt_id + std::to_string( t ) + "s_"
                        + solution_string[static_cast<std::size_t>( Type )]
                        + "_state.csv",
                    { m_x, q1(), q2(), q3() }, { "x", "q1", "q2", "q3" } );
            }
        }

        if ( n_saves > 0 && save_endpoint ) {
            write_to_file<double, Size>(
                opt_id + std::to_string( endpoint ) + "s_"
                    + solution_string[static_cast<std::size_t>( Type )]
                    + "_state.csv",
                { m_x, q1(), q2(), q3() }, { "x", "q1", "q2", "q3" } );
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
                const auto f_i{ get_flux( m_previous_state[i], gamma ) },
                    f_i_1{ get_flux( m_previous_state[i + 1], gamma ) };
                const auto f = flux<T>{ 0.5 * ( f_i + f_i_1 )
                                        + 0.5 * ( m_dx / time_step )
                                              * ( m_previous_state[i]
                                                  - m_previous_state[i + 1] ) };

                return f;
            };

            for ( std::size_t i{ 1 }; i <= Size; ++i ) {
                m_state[i] =
                    m_previous_state[i]
                    + ( time_step / m_dx ) * ( f_half( i - 1 ) - f_half( i ) );
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
    const double gamma = 1.4;

    const auto xarr{ linspace<double, 1000, true>( 0, 1 ) };

    std::array<double, std::tuple_size_v<decltype( xarr )>> q1{};
    std::array<double, std::tuple_size_v<decltype( xarr )>> q2{};
    std::array<double, std::tuple_size_v<decltype( xarr )>> q3{};
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

    fluid_solver<double, std::tuple_size_v<decltype( initial_state )>,
                 solution_type::lax_friedrichs, boundary_type::outflow,
                 boundary_type::outflow>
        fs( 0., 1., initial_state );


    const auto state = fs.simulate( 0.00001, 0.2, gamma, 1, true, "A" );
    /*for ( std::size_t i{ 0 }; i < xarr.size(); ++i ) {
        std::cout << array_string( initial_state[i] ) << "\t"
                  << array_string( state[i + 1] ) << std::endl;
    }*/
}