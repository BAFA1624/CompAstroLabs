#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string_view>
#include <vector>

struct distribution
{
    std::uint64_t              N;
    std::uint64_t              M;
    bool                       weighted;
    std::vector<double>        weights;
    std::vector<std::uint64_t> bins;
    double                     dmin; // Domain minimum
    double                     dmax; // Domain maximum
    double                     rmin; // Range minimum
    double                     rmax; // Range maximum
};

double
draw_sample() {
    static std::uint64_t seed{ 1 };
    seed = ( static_cast<std::uint64_t>( 16807 ) * seed )
           % static_cast<std::uint64_t>( 2147483647 );
    return static_cast<double>( seed ) / static_cast<double>( 2147483647 );
}

distribution
rejection_method( std::function<double( double )> f, std::uint64_t N,
                  std::uint64_t M, const double dmin, const double dmax,
                  const double rmin, const double rmax ) {
    auto bins = std::vector<std::uint64_t>( M, 0. );

    for ( std::uint64_t i = 0; i < N; ++i ) {
        double x{ 0. }, y{ 0. };
        do { // Draw & convert to domain & range of f
            x = ( dmax - dmin ) * draw_sample() + dmin;
            y = ( rmax - rmin ) * draw_sample() + rmin;
        } while ( f( x ) < y );
        // Convert x from domain of f -> 0-1 range.
        x = ( x - dmin ) / ( dmax - dmin );
        // Sort into bin
        bins[static_cast<std::uint64_t>( x * static_cast<double>( M ) )]++;
    }

    const distribution result{ .N = N,
                               .M = M,
                               .weighted = false,
                               .weights = std::vector<double>{},
                               .bins = bins,
                               .dmin = dmin,
                               .dmax = dmax,
                               .rmin = rmin,
                               .rmax = rmax };
    return result;
}

distribution
importance_sampling( const std::function<double( double )> f,
                     const std::uint64_t N, const std::uint64_t M,
                     const double dmin, const double dmax, const double rmin,
                     const double rmax ) {
    // Generate samples
    std::vector<double> samples( N, 0. );
    std::generate( samples.begin(), samples.end(), draw_sample );

    // Count into bins for flat, uniform distribution
    std::vector<std::uint64_t> bins( N, 0 );
    for ( const auto & s : samples ) {
        bins[static_cast<std::uint64_t>( s * static_cast<double>( M ) )]++;
    }

    // Calculate weights
    // First, wi = pi / qi ( wi = weight, pi = f(sample), qi = Q(sample) where Q
    // is the uniform distribution)
    // Then readjust samples
    std::vector<double> weights( M, 0 );
    for ( const auto & s : samples ) {
        const std::uint64_t i =
            static_cast<std::uint64_t>( s * static_cast<double>( M ) );
        // Dividing by rmax scales the flat distribution so Q(x) >= P(x)
        // for all x.
        const auto qi = bins[i] * M / ( rmax * static_cast<double>( N ) );
        // Scale sample into domain of f, then calculate f(x)
        const auto pi = f( ( dmax - dmin ) * s + dmin );

        weights[i] += pi / qi;
    }

    // Reset bins, then recount using weighted samples
    for ( std::uint64_t i = 0; i < M; ++i ) { weights[i] /= bins[i]; }

    const distribution result = { .N = N,
                                  .M = M,
                                  .weighted = true,
                                  .weights = weights,
                                  .bins = bins,
                                  .dmin = dmin,
                                  .dmax = dmax,
                                  .rmin = rmin,
                                  .rmax = rmax };
    return result;
}

double
transform( const distribution & d, const double x ) {
    const double        xval{ ( x - d.dmin ) / ( d.dmax - d.dmin ) };
    const std::uint64_t idx =
        static_cast<std::uint64_t>( xval * static_cast<double>( d.M ) );

    if ( !d.weighted ) {
        return d.bins[idx] * d.M
                   / ( ( d.dmax - d.dmin ) * static_cast<double>( d.N ) )
               + d.rmin;
    }
    else {
        const auto qi =
            d.bins[idx] * d.M / ( d.rmax * static_cast<double>( d.N ) );
        return qi * d.weights[idx];
    }
}

template <typename T>
std::vector<T>
linspace( const T a, const T b, const std::uint64_t N ) {
    const T        delta{ ( b - a ) / N };
    std::vector<T> result;
    auto           tmp{ a };
    for ( std::uint64_t i = 0; i < N; ++i ) {
        result.push_back( tmp );
        tmp += delta;
    }
    return result;
}

void
write_to_file( const std::string_view fname, const std::vector<double> & xvals,
               const std::vector<double> & yvals ) {
    assert( xvals.size() == yvals.size() );

    std::ofstream f( fname, std::ios::out );
    for ( std::uint64_t i = 0; i < xvals.size(); ++i ) {
        f << xvals[i] << "," << yvals[i] << "\n";
    }
}

double
f( const double d ) {
    return ( 2 / M_PI ) * pow( sin( d ), 2.0 );
}

double
g( const double d ) {
    return ( 3. / 8. ) * ( 1 + pow( cos( d ), 2.0 ) ) * sin( d );
}

int
main() {
    const std::uint64_t N = 10000;
    const std::uint64_t M = 100;
    distribution rejection = rejection_method( f, N, M, 0, M_PI, 0, M_2_PI );
    distribution importance =
        importance_sampling( f, 100000, 1000, 0, M_PI, 0, M_2_PI );
    const auto          xvals = linspace( 0., M_PI, 1000 );
    std::vector<double> rejection_vals, importance_vals;
    for ( const auto & x : xvals ) {
        rejection_vals.push_back( transform( rejection, x ) );
        importance_vals.push_back( transform( importance, x ) );
    }

    write_to_file( "rejection_method.csv", xvals, rejection_vals );
    write_to_file( "importance_sampling.csv", xvals, importance_vals );
}