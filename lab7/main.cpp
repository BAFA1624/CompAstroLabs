#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

template <typename T>
concept Numeric = requires( T a ) {
    std::is_integral_v<T> || std::is_floating_point_v<T>;
    !std::is_same_v<T, bool>;
};

template <Numeric T>
std::vector<T>
linspace( const T a, const T b, const std::uint64_t N ) {
    const T        dx{ ( b - a ) / static_cast<T>( N - 1 ) };
    T              tmp{ a };
    std::vector<T> v( N );
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        v[i] = tmp;
        tmp += dx;
    }
    return v;
}

template <Numeric T>
std::vector<T>
arange( const T start, const T end, const T dx ) {
    const std::uint64_t N{ static_cast<std::uint64_t>( ( end - start ) / dx ) };

    T              tmp{ start };
    std::vector<T> v( N );
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
             const T start, const T end, const double h ) {
    approximator<T> approx{};
    switch ( a ) {
    case approx_type::forward: approx = d_fda<double>; break;
    case approx_type::backward: approx = d_bda<double>; break;
    case approx_type::centred: approx = d_cda<double>; break;
    }

    const auto     xvals = arange<T>( start, end, h );
    std::vector<T> yvals( xvals.size() );
    for ( std::uint64_t i{ 0 }; i < xvals.size(); ++i ) {
        yvals[i] = approx( f, xvals[i], h );
    }

    return { xvals, yvals };
}

template <std::floating_point T>
void
write_to_file( const std::string &                 filename,
               const std::vector<std::vector<T>> & values ) {
    std::uint64_t n_rows{ std::numeric_limits<std::uint64_t>::max() };
    for ( const auto & v : values ) {
        n_rows = std::min<std::uint64_t>( n_rows, v.size() );
    }

    std::ofstream fp( filename );
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
                               { fda_xvals, fda_yvals } );
        write_to_file<double>( "bda_" + std::to_string( h ) + ".csv",
                               { bda_xvals, bda_yvals } );
        write_to_file<double>( "cda_" + std::to_string( h ) + ".csv",
                               { cda_xvals, cda_yvals } );
    }
}