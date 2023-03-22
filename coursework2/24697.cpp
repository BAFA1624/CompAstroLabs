#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

enum class distr_type { flat, rejection, weighted };

template <std::floating_point F>
class distribution
{
    public:
    distribution() :
        m_type(),
        m_seed(),
        m_N(),
        m_M(),
        m_bin_count(),
        m_dmin(),
        m_dmax(),
        m_rmin(),
        m_rmax(),
        m_f(),
        m_endpoint(){};

    F transform( const F x ) const noexcept;

    std::vector<F> transform( const std::vector<F> & v ) const noexcept;

    void gen_distribution(
        const distr_type type, const std::uint64_t seed, const std::uint64_t N,
        const std::uint64_t M, const F d_min, const F d_max, const F r_min,
        const F                             r_max,
        const std::function<F( const F )> & f = []( const F ) -> F {
            return 1.;
        },
        const bool include_endpoint = true ) noexcept;

    private:
    F transform_nonweighted( const F x ) const noexcept;
    F transform_weighted( const F x ) const noexcept;

    void initialize_distribution(
        const distr_type type, const std::uint64_t seed, const std::uint64_t N,
        const std::uint64_t M, const F d_min, const F d_max, const F r_min,
        const F                             r_max,
        const std::function<F( const F )> & f = []( const F ) -> F {
            return 1.;
        },
        const bool include_endpoint = true ) noexcept;
    F random_sample() noexcept {
        m_seed = ( static_cast<uint64_t>( 16807 ) * m_seed )
                 % static_cast<std::uint64_t>( 2147483647 );
        return static_cast<F>( m_seed ) / static_cast<F>( 2147483647 );
    }

    distr_type                 m_type;      // Type of distribution
    std::int64_t               m_seed;      // Seed value
    std::uint64_t              m_N;         // Total number of samples
    std::uint64_t              m_M;         // Number of bins
    std::vector<std::uint64_t> m_bin_count; // Sample count per bin
    // Weights for importance sampling, only used for weighted distribution
    std::vector<F> m_weights;
    /* The domain & range min/max. */
    F m_dmin;
    F m_dmax;
    F m_rmin;
    F m_rmax;
    // Function to generate distribution
    std::function<F( const F )> m_f;
    // Is the endpoint included in the distribution
    bool m_endpoint;
};

template <std::floating_point F>
void
distribution<F>::initialize_distribution(
    const distr_type type, const std::uint64_t seed, const std::uint64_t N,
    const std::uint64_t M, const F d_min, const F d_max, const F r_min,
    const F r_max, const std::function<F( const F )> & f,
    const bool include_endpoint ) noexcept {
    m_type = type;
    m_seed = seed;
    m_N = N;
    m_M = M;
    m_bin_count = std::vector<std::uint64_t>( M, 0 );
    if ( m_type == distr_type::weighted ) {
        m_weights = std::vector<F>( m_M, static_cast<F>( 0. ) );
    }
    else {
        m_weights = std::vector<F>{};
    }
    m_dmin = d_min;
    m_dmax = d_max;
    m_rmin = r_min;
    m_rmax = r_max;
    m_f = f;
    m_endpoint = include_endpoint;
}

template <std::floating_point F>
F
distribution<F>::transform( const F x ) const noexcept {
    return ( m_type == distr_type::weighted ) ? transform_weighted( x ) :
                                                transform_nonweighted( x );
}

template <std::floating_point F>
std::vector<F>
distribution<F>::transform( const std::vector<F> & v ) const noexcept {
    std::vector<F> result( v.size() );
    if ( m_type == distr_type::weighted ) {
        std::transform(
            v.cbegin(), v.cend(), result.begin(),
            [this]( const F x ) { return ( *this ).transform_weighted( x ); } );
    }
    else {
        std::transform( v.cbegin(), v.cend(), result.begin(),
                        [this]( const F x ) {
                            return ( *this ).transform_nonweighted( x );
                        } );
    }
    return result;
}

template <std::floating_point F>
F
distribution<F>::transform_nonweighted( const F x ) const noexcept {
    const F dx{ 1 / static_cast<F>( m_M ) };
    const F xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    return ( m_bin_count[static_cast<std::uint64_t>( xval / dx )] * m_M
             / ( ( m_dmax - m_dmin ) * static_cast<F>( m_N ) ) )
           + m_rmin;
}

template <std::floating_point F>
F
distribution<F>::transform_weighted( const F x ) const noexcept {
    const F    xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    const auto i{ static_cast<std::uint64_t>( xval * m_M ) };
    std::cout << "transform_nonweighted(" << x
              << "): " << transform_nonweighted( x ) << ", m_weights[" << i
              << "]: " << m_weights[i] << std::endl;

    return transform_nonweighted( x ) * m_weights[i];
}

template <std::floating_point F>
void
distribution<F>::gen_distribution( const distr_type    type,
                                   const std::uint64_t seed,
                                   const std::uint64_t N, const std::uint64_t M,
                                   const F d_min, const F d_max, const F r_min,
                                   const F                             r_max,
                                   const std::function<F( const F )> & f,
                                   const bool include_endpoint ) noexcept {
    // Fresh initialization, allows gen_distribution to be called repeatedly
    // to generate a new, different distribution.
    initialize_distribution( type, seed, N, M, d_min, d_max, r_min, r_max, f,
                             include_endpoint );

    /* Bin size of distribution to generate
       Distribution is treated as having domain 0 <= x <= 1
       until a new point x is transformed through the distribution. */
    const F dx{ 1 / static_cast<F>( m_M ) };
    const F d_size{ m_dmax - m_dmin };
    const F r_size{ m_rmax - m_rmin };

    switch ( m_type ) {
    case distr_type::flat: {
        for ( std::uint64_t i{ 0 }; i < m_N; ++i ) {
            m_bin_count[static_cast<std::uint64_t>( random_sample() / dx )]++;
        }
    } break;
    case distr_type::rejection: {
        F x{ 0 }, y{ 0 };
        do {
            x = d_size * random_sample() + m_dmin;
            y = r_size * random_sample() + m_rmin;
        } while ( m_f( x ) < y );
        const F xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
        m_bin_count[static_cast<std::uint64_t>( xval / dx )]++;
    } break;
    case distr_type::weighted: {
        //  Generate samples & store for multiple use
        std::vector<F> samples( m_N, F{ 0 } );
        std::generate( samples.begin(), samples.end(),
                       [this]() { return random_sample(); } );

        // Generate flat distribution
        std::for_each( samples.cbegin(), samples.cend(), [this]( const F x ) {
            m_bin_count[static_cast<std::uint64_t>(
                x * static_cast<F>( m_M ) )]++;
        } );

        // Calculate weights in two passes
        // 1) For each sample, weight bin index = same as in flat
        // distribution,
        //    add P(x) / Q(x) to bin
        std::for_each( samples.cbegin(), samples.cend(), [this]( const F x ) {
            const auto i = static_cast<std::uint64_t>(
                static_cast<std::uint64_t>( x * static_cast<F>( m_M ) ) );
            const F qi{ m_bin_count[i] * m_M
                        / ( m_rmax * static_cast<F>( m_N ) ) };
            const F pi{ m_f( x ) };

            m_weights[i] += pi / qi;
        } );
        // 2) For each bin in both the flat distribution and the weights,
        // divide the weight sum by the number of samples in the bin.
        F sum{ 0 };
        for ( std::size_t i = 0; i < m_M; ++i ) {
            m_weights[i] /= m_bin_count[i];
            sum +=
                m_weights[i]
                * ( m_bin_count[i] * m_M / ( m_rmax * static_cast<F>( m_N ) ) );
        }
        std::cout << "sum: " << sum << std::endl;
    } break;
    }

    return;
}

template <typename T>
std::vector<T>
linspace( const T & start, const T & end, const std::uint64_t N,
          const bool include_endpoint = false ) {
    const T dx{
        ( start < end ) ?
            ( end - start ) / static_cast<T>( include_endpoint ? N - 1 : N ) :
            -( start - end ) / static_cast<T>( include_endpoint ? N - 1 : N )
    };

    std::vector<T> result;
    result.reserve( N );
    T tmp{ start };
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        result.emplace_back( tmp );
        tmp += dx;
    }

    return result;
}

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

template <std::floating_point F>
void
write_to_file( const std::string_view fname, const std::vector<F> & x,
               const std::vector<F> & y ) {
    const std::uint64_t n_rows{ std::min( x.size(), y.size() ) };
    std::ofstream       f( fname );
    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        f << x[i] << "," << y[i] << "\n";
    }
}

template <std::floating_point F>
F
mu( const F x ) {
    return cos( x );
}

template <std::floating_point F>
F
p_mu( F m ) {
    return 0.375 * ( 1 + pow( m, 2.0 ) );
}

int
main() {
    const std::uint64_t N{ 10000000 };
    const std::uint64_t M{ 1000 };

    distribution<double> sq_sine{};
    sq_sine.gen_distribution(
        distr_type::weighted, static_cast<std::uint64_t>( 1 ), N, M,
        static_cast<double>( 0 ), static_cast<double>( M_PI ),
        static_cast<double>( 0. ), static_cast<double>( M_2_PI ),
        []( const double x ) -> double {
            return M_2_PI * powl( sinl( x ), 2 );
        } );

    const auto xvals1 = linspace<double>( 0, M_PI, M, false );
    const auto yvals1{ sq_sine.transform( xvals1 ) };

    /*for ( std::uint64_t i = 0; i < M; ++i ) {
        std::cout << "x: " << xvals1[i]
                  << ", f(x): " << M_2_PI * powl( sinl( xvals1[i] ), 2 )
                  << ", y: " << yvals1[i] << std::endl;
    }*/

    write_to_file( "data1.csv", xvals1, yvals1 );

    distribution<double> distr{};
    distr.gen_distribution(
        distr_type::rejection, static_cast<std::uint64_t>( 1 ), N, M,
        static_cast<double>( -1 ), static_cast<double>( 1 ),
        static_cast<double>( 0 ), static_cast<double>( 0.75 ), p_mu<double> );

    const auto test = []( const double m ) -> double {
        return 0.375 * ( 1. + pow( m, 2. ) );
    };
    const auto xvals2 = linspace<double>( -1, 1, M, false );
    const auto yvals2{ distr.transform( xvals2 ) };

    /*for ( std::uint64_t i = 0; i < M; ++i ) {
        std::cout << xvals2[i] << ", " << yvals2[i] << ", " << test(
    xvals2[i] )
                  << std::endl;
    }*/

    write_to_file( "data2.csv", xvals2, yvals2 );
}