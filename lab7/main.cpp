<<<<<<< HEAD
=======
<<<<<<< HEAD
#include <functional>
#include <iostream>

template <typename T>
using Numeric = Concept;

/*deriv_fda( const std::functio ){}*/

int
main() {}
=======
>>>>>>> hand-in
#include <algorithm>
#include <cmath>
#include <concepts>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

template <typename T>
concept Numeric = requires( T a ) {
    std::is_integral_v<T> || std::is_floating_point_v<T>;
    !std::is_same_v<T, bool>;
};

template <class ContainerType>
concept STLContainer = requires( ContainerType x, const ContainerType y ) {
    requires std::regular<ContainerType>;    // Default constructible, copyable,
                                             // & equality comparable
    requires std::swappable<ContainerType>;  // Can be swapped
    requires std::destructible<
        typename ContainerType::value_type>; // Inner values can be destroyed
    // Aliased reference & const_reference are the same as references to the
    // contained type
    requires std::same_as<typename ContainerType::reference,
                          typename ContainerType::value_type &>;
    requires std::same_as<typename ContainerType::const_reference,
                          const typename ContainerType::value_type &>;
    requires std::forward_iterator<typename ContainerType::iterator>;
    requires std::forward_iterator<typename ContainerType::const_iterator>;
    requires std::signed_integral<typename ContainerType::difference_type>;
    requires std::same_as<
        typename ContainerType::difference_type,
        typename std::iterator_traits<
            typename ContainerType::iterator>::difference_type>;
    requires std::same_as<
        typename ContainerType::difference_type,
        typename std::iterator_traits<
            typename ContainerType::const_iterator>::difference_type>;
    { x.begin() } -> std::same_as<typename ContainerType::iterator>;
    { x.end() } -> std::same_as<typename ContainerType::iterator>;
    { y.begin() } -> std::same_as<typename ContainerType::const_iterator>;
    { y.end() } -> std::same_as<typename ContainerType::const_iterator>;
    { x.cbegin() } -> std::same_as<typename ContainerType::const_iterator>;
    { x.cend() } -> std::same_as<typename ContainerType::const_iterator>;
    { x.size() } -> std::same_as<typename ContainerType::size_type>;
    { x.max_size() } -> std::same_as<typename ContainerType::size_type>;
    { x.empty() } -> std::same_as<bool>;
};

/*template <typename Element>
concept RangeElement = requires( Element a, const Element b ) {
    requires std::regular<Element>;
    { a - a } -> std::same_as<Element>;
    { a += a } -> std::same_as<Element>;
    { a += b } -> std::same_as<Element>;
    { a = b } -> std::same_as<Element>;
};

template <typename Container>
concept NumericContainer = requires( Container a ) {
    requires std::regular<Container>;
    requires RangeElement<typename Container::value_type>;
    { a.begin() } -> std::same_as<
};

template <typename ContainerType>
concept NumericContainer = requires( ContainerType a, const ContainerType b ) {
    // Requirements for ContainerType::value_type
    requires Numeric<typename ContainerType::value_type>;
};*/

template <std::floating_point T, STLContainer C>
void
write_to_file( const std::string & filename, const std::vector<C> & values,
               const std::vector<std::string> & keys = {} ) {
    assert( keys.size() == values.size() || keys.size() == 0 );

    std::uint64_t n_rows{ std::numeric_limits<std::uint64_t>::max() };
    for ( const auto & v : values ) {
        n_rows = std::min<std::uint64_t>( n_rows, v.size() );
    }

    std::ofstream fp( filename );
    if ( keys.size() > 0 ) {
        std::string header{ "" };
        for ( const auto & key : keys ) { header += key + ","; }
        header.pop_back();
        header += "\n";
        fp << header;
    }

    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        std::string line{ "" };
        for ( const auto & v : values ) {
            line += std::to_string( v[i] ) + ",";
        }
        line.pop_back();
        line += "\n";
        fp << line;
    }
}

template <typename T, STLContainer C = std::vector<T>>
    requires std::same_as<T, typename C::value_type>
C
linspace( const typename C::value_type a, const typename C::value_type b,
          const std::uint64_t N, const bool endpoint = false ) {
    const T dx{ ( b - a ) / static_cast<T>( endpoint ? N - 1 : N ) };
    T       tmp{ a };
    C       v( N );
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        v[i] = tmp;
        tmp += dx;
    }
    return v;
}

template <Numeric T, std::uint64_t N, bool endpoint = true>
std::array<T, N>
linspace( const T a, const T b ) {
    const T dx{ ( b - a ) / static_cast<T>( ( endpoint ? N - 1 : N ) ) };

    std::array<T, N> result;
    T                tmp{ a };
    for ( auto & x : result ) {
        x = tmp;
        tmp += dx;
    }

    return result;
}

template <typename T>
consteval bool
delta( const T a, const T b ) {
    return a == b;
}

template <Numeric T>
std::vector<T>
arange( const T start, const T end, const T dx ) {
    const std::uint64_t N{ static_cast<std::uint64_t>( ( end - start ) / dx ) };
    T                   tmp{ start };
    std::vector<T>      v( N );
    for ( auto & x : v ) {
        x = tmp;
        tmp += dx;
    }
    return v;
}

template <std::floating_point T>
using approximator =
    std::function<T( const std::function<T( const T )> &, const T, const T )>;

template <std::floating_point T>
T
d_fda( const std::function<T( const T )> & f, const T x, const T h ) {
    return ( f( x + h ) - f( x ) ) / h;
}
template <std::floating_point T>
T
d_bda( const std::function<T( const T )> & f, const T x, const T h ) {
    return ( f( x ) - f( x - h ) ) / h;
}
template <std::floating_point T>
T
d_cda( const std::function<T( const T )> & f, const T x, const T h ) {
    return ( f( x + h ) - f( x - h ) ) / ( 2 * h );
}

enum class approx_type { forward, backward, centred };

template <std::floating_point T>
std::tuple<std::vector<T>, std::vector<T>>
approximate( const approx_type a, const std::function<T( const T )> & f,
             const T start, const T end, const T h ) {
    approximator<T> approx{};
    switch ( a ) {
    case approx_type::forward: approx = d_fda<T>; break;
    case approx_type::backward: approx = d_bda<T>; break;
    case approx_type::centred: approx = d_cda<T>; break;
    }

    const auto xvals = linspace<T>(
        start, end, static_cast<std::uint64_t>( ( end - start ) / h ) );
    std::vector<T> yvals( xvals.size() );
    for ( std::uint64_t i{ 0 }; i < xvals.size(); ++i ) {
        yvals[i] = approx( f, xvals[i], h );
    }

    return { xvals, yvals };
}

/*template <std::floating_point T, std::uint64_t N,
          approx_type A = approx_type::centred, STLContainer C = std::vector<T>>
std::tuple<C, C>
approximate( const std::function<T( const T )> & f, const T start,
             const T end ) {
    approximator<T> approx{};
    switch ( a ) {
    case approx_type::forward: approx = d_fda<T>; break;
    case approx_type::backward: approx = d_bda<T>; break;
    case approx_type::centred: approx = d_cda<T>; break;
    }

    const auto xvals = arange <
}*/

enum class simulation_type {
    diffusive,     // Diffusive scheme
    csdd,          // Centred space derivative difference
    lax_wendroff,  // Lax-Wendroff method
    lax_friedrichs // Lax-Friedrichs method
};
const std::vector<std::string> sim_type_str{ "diffusive", "csdd",
                                             "lax_wendroff", "lax_friedrichs" };

template <std::floating_point T, std::uint64_t N, simulation_type S>
class fluid_simulator
{
    public:
    fluid_simulator( const T x_min, const T x_max,
                     const std::array<T, N> & q_init,
                     const std::array<T, N> & v_init ) :
        m_dx( ( x_max - x_min ) / N ) {
        assert( x_min < x_max );
        m_x = linspace<T, N, true>( x_min, x_max );

        for ( std::uint64_t i{ 0 }; i < N; ++i ) {
            m_q_current[i + 1] = q_init[i];
            m_v_current[i + 1] = v_init[i];
        }

        // Copy values into boundary cells
        m_q_current[0] = m_q_current[1];
        m_q_current[N + 1] = m_q_current[N];

        m_v_current[0] = m_v_current[1];
        m_v_current[N + 1] = m_v_current[N];

        // Set previous state = initial state
        m_q_previous = m_q_current;
        m_v_previous = m_v_current;
    }

    void dx( const T dx_new ) { m_dx = dx_new; }
    void x( const std::array<T, N> & x ) { m_x = x; }
    void q( const std::array<T, N> & q_new ) noexcept {
        std::copy( q_new.cbegin(), q_new.cend(), m_q_current.begin() + 1 );
        m_q_current[0] = m_q_current[1];
        m_q_current[N + 1] = m_q_current[N];
        m_q_previous = m_q_current;
    }
    void v( const std::array<T, N> & v_new ) noexcept {
        std::copy( v_new.cbegin(), v_new.cend(), m_v_current.begin() + 1 );
        m_v_current[0] = m_v_current[1];
        m_v_current[N + 1] = m_v_current[N];
        m_v_previous = m_v_current;
    }
    void state( const std::array<T, N> & q_new,
                const std::array<T, N> & v_new ) noexcept {
        q( q_new );
        v( v_new );
    }

    [[nodiscard]] T                dx() const noexcept { return m_dx; }
    [[nodiscard]] std::array<T, N> x() const noexcept { return m_x; }
    [[nodiscard]] std::array<T, N> q() const noexcept {
        std::array<T, N> q;
        std::copy( m_q_current.cbegin() + 1, m_q_current.cend() - 1,
                   q.begin() );
        return q;
    }
    [[nodiscard]] std::array<T, N> v() const noexcept {
        std::array<T, N> v;
        std::copy( m_v_current.cbegin() + 1, m_v_current.cend() - 1,
                   v.begin() );
        return v;
    }
    [[nodiscard]] std::tuple<std::array<T, N>, std::array<T, N>>
    state() const noexcept {
        return { q(), v() };
    }

    void simulate( const T end_time, const T time_step,
                   const std::uint64_t n_saves,
                   const bool          endpoint = true ) noexcept;

    private:
    void update_q( const T dt ) noexcept;

    T                    m_dx;
    std::array<T, N>     m_x;
    std::array<T, N + 2> m_q_current;
    std::array<T, N + 2> m_q_previous;
    std::array<T, N + 2> m_v_current;
    std::array<T, N + 2> m_v_previous;
};

template <std::floating_point T, std::uint64_t N, simulation_type S>
void
fluid_simulator<T, N, S>::update_q( const T dt ) noexcept {
    if constexpr ( S == simulation_type::diffusive ) {
        for ( std::uint64_t i{ 0 }; i < N; ++i ) {
            m_q_current[i + 1] = -( m_v_previous[i + 1] * dt / m_dx )
                                     * ( m_q_previous[i + 1] - m_q_previous[i] )
                                 + m_q_previous[i + 1];
        }
    }
    else if constexpr ( S == simulation_type::csdd ) {
        for ( std::uint64_t i{ 0 }; i < N; ++i ) {
            m_q_current[i + 1] = -0.5 * ( m_v_previous[i + 1] * dt / m_dx )
                                     * ( m_q_previous[i + 2] - m_q_previous[i] )
                                 + m_q_previous[i + 1];
        }
    }
    else if constexpr ( S == simulation_type::lax_wendroff ) {
        for ( std::uint64_t i{ 0 }; i < N; ++i ) {
            m_q_current[i + 1] =
                m_q_previous[i + 1]
                - ( 0.5 * m_v_previous[i + 1] * dt / m_dx )
                      * ( m_q_previous[i + 2] - m_q_previous[i] )
                + ( 0.5 * m_v_previous[i + 1] * m_v_previous[i + 1] * dt * dt
                    / ( m_dx * m_dx ) )
                      * ( m_q_previous[i + 2] - 2 * m_q_previous[i + 1]
                          + m_q_previous[i] );
        }
    }
    else if constexpr ( S == simulation_type::lax_friedrichs ) {
        const auto f_half = [*this, &dt]( const std::uint64_t i ) {
            return 0.5
                       * ( m_q_previous[i] * m_v_previous[i]
                           + m_q_previous[i + 1] * m_v_previous[i + 1] )
                   + 0.5 * m_dx * ( m_q_previous[i] - m_q_previous[i + 1] )
                         / dt;
        };
        for ( std::uint64_t i{ 0 }; i < N; ++i ) {
            m_q_current[i + 1] =
                m_q_previous[i + 1]
                + dt * ( f_half( i ) - f_half( i + 1 ) ) / m_dx;
        }
    }

    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        m_x[i] += m_v_current[i + 1] * dt;
    }

    return;
}

template <std::floating_point T, std::uint64_t N, simulation_type S>
void
fluid_simulator<T, N, S>::simulate( const T end_time, const T time_step,
                                    const std::uint64_t n_saves,
                                    const bool          endpoint ) noexcept {
    const auto n_steps{ static_cast<std::uint64_t>( ( end_time )
                                                    / time_step ) };
    const auto save_state_freq{ n_steps
                                / ( endpoint ? n_saves - 1 : n_saves ) };

    for ( T t{ 0 }; t <= end_time; t += time_step ) {
        update_q( time_step );

        m_q_current[0] = m_q_current[1];
        m_q_current[N + 1] = m_q_current[N];

        m_v_current[0] = m_v_current[1];
        m_v_current[N + 1] = m_v_current[N];

        m_q_previous = m_q_current;
        m_v_previous = m_v_current;

        const auto i{ static_cast<std::uint64_t>( t / time_step ) };
        if ( i % save_state_freq == 0 ) {
            write_to_file<T>(
                std::to_string( t ) + "s_"
                    + sim_type_str[static_cast<std::uint64_t>( S )]
                    + "_state.csv",
                std::vector{ x(), q(), v() }, { "x", "q", "v" } );
        }
    }
    if ( n_saves > 0 && endpoint ) {
        write_to_file<T>( std::to_string( end_time ) + "s_"
                              + sim_type_str[static_cast<std::uint64_t>( S )]
                              + "_state.csv",
                          std::vector{ x(), q(), v() }, { "x", "q", "v" } );
    }
    return;
}

template <STLContainer C>
void
f( const C & x ) {
    std::cout << x.size() << std::endl;
}

template <Numeric T>
consteval T
add( const T a, const T b ) {
    return a + b;
}

int
main() {
    const double start{ 0.0 };
    const double end{ 10.0 };
    const auto   stepsize = std::vector<double>{ 1, 0.1, 0.01, 0.001, 0.0001 };
    for ( const auto & h : stepsize ) {
        const auto [fda_xvals, fda_yvals] = approximate<double>(
            approx_type::forward,
            []( const double x ) { return std::exp( x ); }, start, end, h );
        const auto [bda_xvals, bda_yvals] = approximate<double>(
            approx_type::backward,
            []( const double x ) { return std::exp( x ); }, start, end, h );
        const auto [cda_xvals, cda_yvals] = approximate<double>(
            approx_type::centred,
            []( const double x ) { return std::exp( x ); }, start, end, h );

        write_to_file<double>( "fda_" + std::to_string( h ) + ".csv",
                               std::vector{ fda_xvals, fda_yvals } );
        write_to_file<double>( "bda_" + std::to_string( h ) + ".csv",
                               std::vector{ bda_xvals, bda_yvals } );
        write_to_file<double>( "cda_" + std::to_string( h ) + ".csv",
                               std::vector{ cda_xvals, cda_yvals } );
    }

    const double xmin{ 0 };
    const double xmax{ 400 };

    std::array<double, 4000> init_v;
    init_v.fill( 100. );

    std::array<double, std::tuple_size_v<decltype( init_v )>> init_q;

    const double dx{ ( xmax - xmin )
                     / std::tuple_size<decltype( init_q )>::value };
    for ( std::uint64_t i{ 0 }; i < std::tuple_size<decltype( init_q )>::value;
          ++i ) {
        const double x{ i * dx };
        if ( x < 10 ) {
            init_q[i] = 1.;
        }
        else if ( x < 50 ) {
            init_q[i] = 0.5;
        }
        else {
            init_q[i] = 0.;
        }
    }

    const double        end_time{ 3. };
    const double        dt{ 0.00002 };
    const std::uint64_t n_saves{ 4 };

    fluid_simulator<double, std::tuple_size_v<decltype( init_q )>,
                    simulation_type::diffusive>
        diffusive_fs( xmin, xmax, init_q, init_v );
    diffusive_fs.simulate( end_time, dt, n_saves );

    fluid_simulator<double, std::tuple_size_v<decltype( init_q )>,
                    simulation_type::csdd>
        csdd_fs( xmin, xmax, init_q, init_v );
    csdd_fs.simulate( end_time, dt, n_saves );

    fluid_simulator<double, std::tuple_size_v<decltype( init_q )>,
                    simulation_type::lax_wendroff>
        lax_wendroff_fs( xmin, xmax, init_q, init_v );
    lax_wendroff_fs.simulate( end_time, dt, n_saves );

    fluid_simulator<double, std::tuple_size_v<decltype( init_q )>,
                    simulation_type::lax_friedrichs>
        lax_friedrichs_fs( xmin, xmax, init_q, init_v );
    lax_friedrichs_fs.simulate( end_time, dt, n_saves );
<<<<<<< HEAD
}
=======
}
>>>>>>> coursework3_array
>>>>>>> hand-in
