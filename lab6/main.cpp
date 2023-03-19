#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T,
          typename T_it = decltype( std::begin( std::declval<T>() ) )>
constexpr auto
enumerate( const T & iterable ) {
    struct enum_it
    {
        enum_it( const std::size_t i, const T_it it ) : m_i( i ), m_it( it ) {}

        [[nodiscard]] bool operator!=( const enum_it & other ) const noexcept {
            return m_it != other.m_it;
        }
        void operator++() noexcept {
            ++m_i;
            ++m_it;
        }
        auto operator*() const { return std::tie( m_i, *m_it ); }
        auto operator*() { return std::tie( m_i, *m_it ); }

        std::size_t m_i;
        T_it        m_it;
    };
    struct enum_it_wrapper
    {
        T m_it;

        enum_it_wrapper( const T & it ) : m_it( it ) {}

        [[nodiscard]] auto begin() const noexcept {
            return enum_it( 0, std::begin( m_it ) );
        }
        [[nodiscard]] auto cbegin() const {
            return enum_it( 0, std::cbegin( m_it ) );
        }
        [[nodiscard]] auto end() const noexcept {
            return enum_it{ 0, std::end( m_it ) };
        }
        [[nodiscard]] auto cend() const noexcept {
            return enum_it{ 0, std::cend( m_it ) };
        }
    };
    return enum_it_wrapper( iterable );
}

// Helper functions and types

// Need to select the correct const/non-const iterator type
// for a given element in each tuple.
// 1) Check if const on reference-stripped element.
// 2) Return the iterator type of the decayed type.
template <typename T>
using select_iter_for =
    std::conditional_t<std::is_const_v<std::remove_reference_t<T>>,
                       typename std::decay_t<T>::const_iterator,
                       typename std::decay_t<T>::iterator>;
template <typename Iter>
using select_access_type = std::conditional_t<
    std::is_same_v<Iter, std::vector<bool>::iterator>
        || std::is_same_v<Iter, std::vector<bool>::const_iterator>,
    typename std::iterator_traits<Iter>::value_type,
    typename std::iterator_traits<Iter>::reference>;

template <typename... Ts, std::size_t... Index>
auto
any_match_impl( const std::tuple<Ts...> & lhs, const std::tuple<Ts...> & rhs,
                std::index_sequence<Index...> ) -> bool {
    return ( ... | ( std::get<Index>( lhs ) == std::get<Index>( rhs ) ) );
}
template <typename... Ts>
auto
any_match( const std::tuple<Ts...> & lhs, const std::tuple<Ts...> & rhs )
    -> bool {
    return any_match_impl( lhs, rhs, std::index_sequence_for<Ts...>{} );
}

template <typename... Iters>
class zip_iter
{
    public:
    using m_value_type = std::tuple<select_access_type<Iters>...>;

    zip_iter() = delete;
    zip_iter( Iters &&... iters ) :
        m_iters{ std::forward<Iters>( iters )... } {}

    auto operator++() -> zip_iter & {
        std::apply( []( auto &... args ) { ( ( args += 1 ), ... ); }, m_iters );
        return *this;
    }
    auto operator++( int ) -> zip_iter {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    [[nodiscard]] auto operator==( const zip_iter & other ) {
        return any_match( m_iters, other.m_iters );
    }
    [[nodiscard]] auto operator!=( const zip_iter & other ) {
        return !( *this == other ); //! any_match( m_iters, other.m_iters );
    }

    [[nodiscard]] auto operator*() {
        return std::apply(
            []( auto &&... args ) { return m_value_type( *args... ); },
            m_iters );
    }

    private:
    std::tuple<Iters...> m_iters;
};
template <typename... T>
class zipper
{
    public:
    using m_zip_t = zip_iter<select_iter_for<T>...>;

    template <typename... Args>
    zipper( Args &&... args ) : m_args( std::forward<Args>( args )... ) {}

    [[nodiscard]] auto begin() -> m_zip_t {
        return std::apply(
            []( auto &&... args ) { return m_zip_t( std::begin( args )... ); },
            m_args );
    }
    [[nodiscard]] auto end() -> m_zip_t {
        return std::apply(
            []( auto &&... args ) { return m_zip_t( std::end( args )... ); },
            m_args );
    }

    private:
    std::tuple<T...> m_args;
};

template <typename... Ts>
constexpr auto
zip( Ts &&... args ) {
    return zipper<Ts...>{ std::forward<Ts>( args )... };
}

template <typename T>
std::vector<T>
linspace( T a, const T b, const std::size_t N, const bool endpoint_inclusive ) {
    const T h =
        ( a < b ) ?
            ( b - a ) / static_cast<T>( endpoint_inclusive ? N - 1 : N ) :
            -( a - b ) / static_cast<T>( endpoint_inclusive ? N - 1 : N );
    std::vector<T> v( N );
    // clang-format off
    std::generate( v.begin(), v.end(),
        [x = a, &h]() mutable {
            const T tmp = x;
            x += h;
            return tmp;
        }
    );
    // clang-format on
    return v;
}

template <typename T>
auto
arange( T a, const T b, const std::size_t N ) {
    std::vector<T> v( N );
    // clang-format off
    std::generate( v.begin(), v.end(),
        [x = a, &b]() mutable {
            const auto tmp = x;
            x += b;
            return tmp;
        }
    );
    // clang-format on
    return v;
}

void
print( const std::string & fname, const std::vector<double> & s_vals,
       const std::vector<double> & i_vals ) {
    std::fstream fp;
    fp.open( fname, std::ios::out );

    for ( const auto & [s, i] : zip( s_vals, i_vals ) ) {
        fp << std::to_string( s ) + "," + std::to_string( i ) + "\n";
    }

    fp.close();
}

void
plot( const std::size_t & N, const std::string & title,
      const std::string & fname, const std::vector<std::string> & fnames,
      const std::vector<std::vector<double>> & s_vals,
      const std::vector<std::vector<double>> & i_vals ) {
    FILE * plot_pipe = popen( "gnuplot -persist", "w" );
    if ( !plot_pipe ) {
        std::cout << "Failed to open pipe for gnuplot.\n";
        exit( EXIT_FAILURE );
    }

    fprintf( plot_pipe,
             "set terminal png size 1200,900 enhanced font 'Ariel'\n" );
    fprintf( plot_pipe, "set output '%s.png'\n", fname.c_str() );
    fprintf( plot_pipe, "set title '%s'\n", title.c_str() );
    fprintf( plot_pipe, "set xlabel 's'\n" );
    fprintf( plot_pipe, "set ylabel 'I'\n" );
    fprintf( plot_pipe, "set datafile separator ','\n" );

    std::string plot_cmd{ "plot " };
    for ( auto i{ 0 }; i < s_vals.size() - 1; ++i ) {
        plot_cmd += "'" + fnames[i] + ".txt' using " + "1:2" + " title '"
                    + std::to_string( s_vals[i].back() ) + "' with lines, ";
    }
    plot_cmd += "'" + fnames.back() + ".txt' using " + "1:2" + " title '"
                + std::to_string( s_vals.back().back() ) + "' with lines";

    fprintf( plot_pipe, "%s\n", plot_cmd.c_str() );

    fflush( plot_pipe );

    pclose( plot_pipe );
}


int
main() {
    constexpr static auto smin = 0.;
    constexpr static auto smax = 100.;

    constexpr static auto eta1 = 3;
    constexpr static auto eta2 = 2;

    constexpr static auto alpha_v = []( const auto & s, const auto & s_max ) {
        return pow( s / s_max, eta1 );
    };
    constexpr static auto j_v = []( const auto & s, const auto & s_max ) {
        return pow( s / s_max, eta2 );
    };
    constexpr static auto tau = []( const auto & s, const auto & s_max ) {
        return s * alpha_v( s, s_max ) / ( eta1 + 1 );
    };
    constexpr static auto I_j = [&]( const auto & I_prev, const auto & s_j,
                                     const auto & s_prev, const auto & s_max ) {
        return I_prev
               - alpha_v( s_j, s_max ) * ( I_prev - s_prev ) * ( s_j - s_prev );
    };

    const std::size_t         n_grid{ 100 };
    const std::vector<double> smax_rng =
        arange( static_cast<double>( 30 ), static_cast<double>( 30 ), 3 );
    const std::vector<std::size_t> N_rng = arange(
        static_cast<std::size_t>( 2 ), static_cast<std::size_t>( 2 ), 6 );


    std::cout << "Q2.a:" << std::endl;
    for ( const auto & N : N_rng ) {
        std::vector<std::vector<double>> s_arr;
        std::vector<std::vector<double>> i_arr;
        for ( const auto & smax : smax_rng ) {
            std::vector<std::size_t>  N_vals;
            const std::vector<double> s = linspace( smin, smax, N, false );
            s_arr.push_back( s );

            std::vector<double> i( N );
            i[0] = 0;
            for ( std::size_t j = 1; j < N; ++j ) {
                i[j] = I_j( i[j - 1], s[j], s[j - 1], s.back() );
            }

            i_arr.push_back( i );
        }

        std::vector<std::string> fnames;
        for ( const auto & [s, i] : zip( s_arr, i_arr ) ) {
            const std::string fname = "2a_N_" + std::to_string( N ) + "_smax_"
                                      + std::to_string( i.back() );
            print( fname + ".txt", s, i );
            fnames.push_back( fname );
        }

        plot( N, "2.a - N = " + std::to_string( N ),
              "2a_N_" + std::to_string( N ), fnames, s_arr, i_arr );
    }

    std::cout << "Q2.b:" << std::endl;

    constexpr auto I_b = []( const double & I_a, const double & s_b,
                             const double & s_a, const double & s_max ) {
        const double t_a = tau( s_a, s_max );
        const double t_b = tau( s_b, s_max );
        const double dt = t_b - t_a;
        const double e_dt = exp( -dt );
        const double x = ( 1 - e_dt ) / dt;
        return I_a * e_dt + s_a * ( x - e_dt ) + s_b * ( 1 - x );
    };

    for ( const auto & N : N_rng ) {
        std::vector<std::vector<double>> s_arr;
        std::vector<std::vector<double>> i_arr;
        for ( const auto & smax : smax_rng ) {
            std::vector<std::size_t>  N_vals;
            const std::vector<double> s = linspace( smin, smax, N, false );
            s_arr.push_back( s );

            std::vector<double> i( N );
            i[0] = 0;
            for ( std::size_t j = 1; j < N; ++j ) {
                i[j] = I_b( i[j - 1], s[j], s[j - 1], smax );
            }

            i_arr.push_back( i );
        }

        std::vector<std::string> fnames;
        for ( const auto & [s, i] : zip( s_arr, i_arr ) ) {
            const std::string fname = "2b_N_" + std::to_string( N ) + "_smax_"
                                      + std::to_string( i.back() );
            print( fname + ".txt", s, i );
            fnames.push_back( fname );
        }

        plot( N, "2.b - N = " + std::to_string( N ),
              "2b_N_" + std::to_string( N ), fnames, s_arr, i_arr );
    }
}