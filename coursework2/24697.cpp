#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
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

    F random_sample() noexcept {
        m_seed = ( static_cast<std::uint64_t>( 16807 ) * m_seed )
                 % static_cast<std::uint64_t>( 2147483647 );
        return static_cast<F>( m_seed ) / static_cast<F>( 2147483647 );
    }

    F random() const noexcept {
        const F x{ ( m_dmax - m_dmin ) * random_sample() + m_dmin };
        return transform( x );
    }

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
    const F xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    return ( ( m_bin_count[static_cast<std::uint64_t>(
                   xval * static_cast<F>( m_M ) )]
               * m_M / ( ( m_dmax - m_dmin ) * static_cast<F>( m_N ) ) )
             + m_rmin );
}

template <std::floating_point F>
F
distribution<F>::transform_weighted( const F x ) const noexcept {
    const F    xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    const auto i{ static_cast<std::uint64_t>( xval * m_M ) };
    return m_weights[i] * m_bin_count[i] * m_M
           / ( m_rmax * static_cast<F>( m_N ) );
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
    const F d_size{ m_dmax - m_dmin };
    const F r_size{ m_rmax - m_rmin };

    switch ( m_type ) {
    case distr_type::flat: {
        for ( std::uint64_t i{ 0 }; i < m_N; ++i ) {
            m_bin_count[static_cast<std::uint64_t>(
                random_sample() * static_cast<F>( m_M ) )]++;
        }
    } break;
    case distr_type::rejection: {
        for ( std::uint64_t i{ 0 }; i < m_N; ++i ) {
            F x{ 0 }, y{ 0 };
            do {
                x = d_size * random_sample() + m_dmin;
                y = r_size * random_sample() + m_rmin;
            } while ( m_f( x ) < y );
            const F xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
            m_bin_count[static_cast<std::uint64_t>(
                xval * static_cast<F>( m_M ) )]++;
        }
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

        // Draw N new samples
        std::vector<F> new_samples( m_N, F{ 0 } );
        std::generate( new_samples.begin(), new_samples.end(),
                       [this]() { return random_sample(); } );

        // Calculate weights in two passes
        // 1) For each sample, weight bin index = same as in flat
        // distribution,
        //    add P(x) / Q(x) to bin
        for ( const auto & s : new_samples ) {
            const auto i{ static_cast<std::uint64_t>(
                s * static_cast<F>( m_M ) ) };
            const F    xval = ( m_dmax - m_dmin ) * s + m_dmin;
            const F    p_flat{ m_bin_count[i] * m_M / ( ( m_rmax ) *m_N ) };
            m_weights[i] += ( m_f( xval ) / p_flat );
        }
        // 2) For each bin in both the flat distribution and the weights,
        // divide the weight sum by the number of samples in the bin.
        for ( std::uint64_t i{ 0 }; i < m_M; ++i ) {
            m_weights[i] /= m_bin_count[i];
        }

        break;
    }

        return;
    }
}


enum class angle_change_type { add, random };

template <std::floating_point F>
class photon
{
    public:
    photon( const angle_change_type change_type, const F x, const F y,
            const F z, const F theta, const F phi ) :
        m_change_type( change_type ),
        m_x( x ),
        m_y( y ),
        m_z( z ),
        m_theta( theta ),
        m_phi( phi ),
        m_absorbed( false ) {}

    photon & move( const F tau = 0 ) noexcept;
    photon & scatter( const F theta = 0, const F phi = 0 ) noexcept;
    // photon & scatter_add( const F theta, const F phi ) noexcept;
    // photon & scatter_random( const F theta, const F phi ) noexcept;

    [[nodiscard]] std::tuple<F, F, F> pos() const noexcept {
        return { m_x, m_y, m_z };
    }
    [[nodiscard]] std::tuple<F, F> dir() const noexcept {
        return { m_theta, m_phi };
    }
    [[nodiscard]] F x() const noexcept { return m_x; }
    [[nodiscard]] F y() const noexcept { return m_y; }
    [[nodiscard]] F z() const noexcept { return m_z; }
    [[nodiscard]] F theta() const noexcept { return m_theta; }
    [[nodiscard]] F phi() const noexcept { return m_phi; }
    [[nodiscard]] F absorbed() const noexcept { return m_absorbed; }
    void            set_absorbed() noexcept { m_absorbed = true; }

    private:
    F m_x;
    F m_y;
    F m_z;

    F m_theta;
    F m_phi;

    bool m_absorbed;

    angle_change_type m_change_type;
};

template <std::floating_point F>
photon<F> &
photon<F>::move( const F tau ) noexcept {
    const F s_theta{ std::sin( m_theta ) }, c_theta{ std::cos( m_theta ) },
        s_phi{ std::sin( m_phi ) }, c_phi{ std::cos( m_phi ) };
    m_x += tau * s_theta * c_phi;
    m_y += tau * s_theta * s_phi;
    m_z += tau * c_theta;

    return *this;
}

template <std::floating_point F>
photon<F> &
photon<F>::scatter( const F theta, const F phi ) noexcept {
    constexpr if ( m_change_type == angle_change_type::random ) {
        m_theta = theta;
        m_phi = phi;
    }
    else {
        const F tmp_theta = m_theta + theta;
        m_theta = tmp_theta + std::floor( tmp_theta / M_PI ) * ( -2 * theta );
        m_phi = ( m_phi + phi ) % ( 2 * M_PI );
    }
    // std::cout << "theta: " << m_theta << std::endl;
    // std::cout << "phi: " << m_phi << std::endl;
    return *this;
}

template <std::floating_point F>
photon<F>
track_photon( photon<F> & p, const distribution<F> & d_theta,
              const distribution<F> & d_phi, const distribution<F> & d_albedo,
              const F tau = 10, const F albedo = 1., const F zmin = 0.,
              const F zmax = 1. ) {
    goto LOOP_START;

REGENERATE_PHOTON:
    p = photon<F>{ type, 0, 0, 0, 0, 0 };
    goto LOOP_START;

    do {
        // Scatter or absorb?
        if ( albedo.random_sample() >= a ) {
            // Absorb photon
            std::cout << "Photon absorbed, this is shouldn't be reached."
                      << std::endl;
            p.set_absorbed();
            goto END;
        }

        // Scatter
        p.scatter( theta.random(), phi.random() );

        // Move
    LOOP_START:
        p.move( tau );
    } while ( p.z() >= zmin && p.z() <= zmax );

    if ( z < zmin ) {
        goto REGENERATE_PHOTON;
    }

END:
    return p;
}

template <std::floating_point F>
void
isotropic_scattering( const std::uint64_t N, const angle_change_type type,
                      const F tau = 10, const F albedo = 1, const F zmin = 0.,
                      const F zmax = 0 ) {
    distribution<F> d_theta;
    distribution<F> d_phi;
    distribution<F> d_albedo;
    // Initialize theta & phi distributions with some random seed values
    d_theta.gen_distribution( distr_type::flat, 12341235, 1000000, 1000, 0, 1,
                              0, M_PI );
    d_phi.gen_distribution( distr_type::flat, 5439867, 1000000, 1000, 0, 1, 0,
                            M_PI );
    // Albedo distribution doesn't need high N, the distribution needs to vary
    // from 0 - 1 so just use result of distribution::random_sample() instead.
    d_albedo.gen_distribution( distr_type::flat, 1, 1, 1, 0, 1, 0, 1 );

    // Initialize N photons at position (0, 0, 0) and angle (0, 0)
    std::vector<photon<F>> photons( N, photon<F>{ type, 0, 0, 0, 0, 0 } );

    // Launch particles
    std::for_each( photons.begin(), photons.end(), [&]( const photon<F> & p ) {
        return track_photon( p, d_theta, d_phi, d_albedo, tau, albedo, zmin,
                             zmax );
    } );

    // Bin by intensity
    const F d_mu{ 2 * M_PI / 10 };
}

void
thomson_scattering() {}


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

template <std::floating_point F>
void
write_to_file( const std::string_view fname, const std::vector<F> & x,
               const std::vector<F> & y, const auto precision ) {
    const std::uint64_t n_rows{ std::min( x.size(), y.size() ) };
    std::ofstream       f( fname );
    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        f << std::setprecision( precision ) << x[i] << "," << y[i] << "\n";
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

template <std::floating_point F>
F
p_theta( F theta ) {
    return 0.375 * ( 1 + pow( cos( theta ), 2.0 ) ) * sin( theta );
}

int
main() {
    const std::uint64_t N{ 1000000 };
    const std::uint64_t M{ 1000 };

    const double dmin{ -1 }, dmax{ 1 }, rmin{ 0. }, rmax{ 0.75 };

    distribution<double> rejection{};
    rejection.gen_distribution( distr_type::rejection,
                                static_cast<std::uint64_t>( 1 ), N, M, dmin,
                                dmax, rmin, rmax, p_mu<double> );

    const auto xvals1 = linspace<double>( dmin, dmax, M, false );
    const auto yvals1{ rejection.transform( xvals1 ) };

    write_to_file( "data1.csv", xvals1, yvals1, 20 );

    distribution<double> weighted{};
    weighted.gen_distribution( distr_type::weighted,
                               static_cast<std::uint64_t>( 1 ), N, M, dmin,
                               dmax, rmin, rmax, p_mu<double> );

    const auto xvals2 = linspace<double>( dmin, dmax, M, false );
    const auto yvals2{ weighted.transform( xvals2 ) };

    write_to_file( "data2.csv", xvals2, yvals2, 20 );
}