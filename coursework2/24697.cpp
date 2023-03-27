#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>


template <typename T>
concept Numeric = requires( T x ) {
    requires std::is_integral_v<T> || std::is_floating_point_v<T>;
    requires !std::is_same_v<bool, T>;
};

template <std::floating_point F>
F
simpson_3_8( const std::function<F( const F )> & f, const F start, const F end,
             const std::uint64_t N ) {
    F       sum{ 0. };
    const F dx{ ( end - start ) / N };
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        const F a{ i * dx };
        const F b{ ( i + 1 ) * dx };
        sum += 0.125 * ( b - a )
               * ( f( a ) + 3 * f( ( 2 * a + b ) / 3 )
                   + 3 * f( ( a + 2 * b ) / 3 ) + f( b ) );
    }
    return sum;
}

template <Numeric T>
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

template <Numeric T>
std::vector<T>
cumulative_sum( const typename std::vector<T>::const_iterator & cbegin,
                const typename std::vector<T>::const_iterator & cend ) {
    std::vector<T> result( std::distance( cbegin, cend ) );

    T sum{ 0 };
    std::transform( cbegin, cend, result.begin(), [&sum]( const T & x ) -> T {
        sum += x;
        return sum;
    } );

    return result;
}

template <Numeric T>
double
MSE( const std::vector<T> & predicted_values,
     const std::vector<T> & true_values ) {
    assert( predicted_values.size() == true_values.size() );
    T sum{ 0 };
    for ( std::uint64_t i{ 0 }; i < predicted_values.size(); ++i ) {
        sum += ( predicted_values[i] - true_values[i] )
               * ( predicted_values[i] - true_values[i] );
    }
    return sum / predicted_values.size();
}

std::uint64_t
random( const std::uint64_t s, const bool set_seed = false ) {
    static std::uint64_t seed{ 0 };
    if ( set_seed ) {
        seed = s;
    }

    seed = ( seed * static_cast<std::uint64_t>( 16807 ) )
           % static_cast<std::uint64_t>( 2147483647 );

    return seed;
}

template <std::floating_point F>
F
random_f( const std::uint64_t s, const bool set_seed = false ) {
    static std::uint64_t seed{ 0 };
    if ( set_seed ) {
        seed = s;
    }

    seed = ( seed * static_cast<std::uint64_t>( 16807 ) )
           % static_cast<std::uint64_t>( 2147483647 );

    return static_cast<F>( seed ) / static_cast<F>( 2147483647 );
}

template <Numeric T1, Numeric T2>
void
write_to_file( const std::string & fname, const std::vector<T1> & x,
               const std::vector<T2> & y, const auto precision ) {
    const std::uint64_t n_rows{ std::min( x.size(), y.size() ) };
    std::ofstream       f( fname, std::ios_base::out );
    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        f << std::setprecision( precision ) << x[i] << "," << y[i] << "\n";
    }
}


enum class distr_type { flat, rejection, importance, cumulative };

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
        m_cnorm(),
        m_f(),
        m_f_inverse() {}

    F random_sample() noexcept {
        m_seed = ( static_cast<std::uint64_t>( 16807 ) * m_seed )
                 % static_cast<std::uint64_t>( 2147483647 );
        return static_cast<F>( m_seed ) / static_cast<F>( 2147483647 );
    }

    F random() noexcept {
        if ( m_type == distr_type::cumulative ) {
            return m_cnorm * m_f_inverse( random_sample() );
        }
        else {
            return ( m_rmax - m_rmin ) * random_sample() + m_rmin;
        }
    }

    std::vector<F> random( const std::uint64_t N ) noexcept {
        std::vector<F> result( N );
        std::generate( result.begin(), result.end(),
                       [this]() { return random(); } );
        return result;
    }

    F transform( const F x ) const noexcept;

    std::vector<F> transform( const std::vector<F> & v ) const noexcept;

    void gen_distribution(
        const distr_type type, const std::uint64_t seed, const std::uint64_t N,
        const std::uint64_t M, const F d_min, const F d_max, const F r_min,
        const F                             r_max,
        const std::function<F( const F )> & f = []( const F ) -> F {
            return static_cast<F>( 0 );
        },
        const std::function<F( const F )> & f_inverse = []( const F ) -> F {
            return static_cast<F>( 0 );
        } ) noexcept;

    private:
    F transform_nonweighted( const F x ) const noexcept;
    F transform_importance( const F x ) const noexcept;
    F transform_cumulative( const F x ) const noexcept;

    void initialize_distribution(
        const distr_type type, const std::uint64_t seed, const std::uint64_t N,
        const std::uint64_t M, const F d_min, const F d_max, const F r_min,
        const F r_max, const std::function<F( const F )> & f,
        const std::function<F( const F )> & f_inverse ) noexcept;


    distr_type                 m_type;      // Type of distribution
    std::int64_t               m_seed;      // Seed value
    std::uint64_t              m_N;         // Total number of samples
    std::uint64_t              m_M;         // Number of bins
    std::vector<std::uint64_t> m_bin_count; // Sample count per bin
    // Weights for importance sampling, only used for importance sampled
    // distribution
    std::vector<F> m_weights;
    /* The domain & range min/max. */
    F m_dmin;
    F m_dmax;
    F m_rmin;
    F m_rmax;
    // Normalization constant to determined for the cumulative method
    F m_cnorm;
    // Function to generate distribution
    std::function<F( const F )> m_f;
    std::function<F( const F )> m_f_inverse;
};

template <std::floating_point F>
void
distribution<F>::initialize_distribution(
    const distr_type type, const std::uint64_t seed, const std::uint64_t N,
    const std::uint64_t M, const F d_min, const F d_max, const F r_min,
    const F r_max, const std::function<F( const F )> & f,
    const std::function<F( const F )> & f_inverse ) noexcept {
    m_type = type;
    m_seed = seed;
    m_N = N;
    m_M = M;
    m_bin_count = std::vector<std::uint64_t>( M, 0 );
    if ( m_type == distr_type::importance ) {
        m_weights = std::vector<F>( m_M, static_cast<F>( 0. ) );
    }
    else {
        m_weights = std::vector<F>{};
    }
    m_dmin = d_min;
    m_dmax = d_max;
    m_rmin = r_min;
    m_rmax = r_max;
    m_cnorm = 1.;
    m_f = f;
    m_f_inverse = f_inverse;
}

template <std::floating_point F>
F
distribution<F>::transform( const F x ) const noexcept {
    return ( m_type == distr_type::importance ) ? transform_importance( x ) :
                                                  transform_nonweighted( x );
}

template <std::floating_point F>
std::vector<F>
distribution<F>::transform( const std::vector<F> & v ) const noexcept {
    std::vector<F> result( v.size() );
    switch ( m_type ) {
    case distr_type::importance: {
        std::transform( v.cbegin(), v.cend(), result.begin(),
                        [this]( const F x ) {
                            return ( *this ).transform_importance( x );
                        } );
    } break;
    case distr_type::cumulative: {
        std::transform( v.cbegin(), v.cend(), result.begin(),
                        [this]( const F x ) {
                            return ( *this ).transform_cumulative( x );
                        } );
    } break;
    default: {
        std::transform( v.cbegin(), v.cend(), result.begin(),
                        [this]( const F x ) {
                            return ( *this ).transform_nonweighted( x );
                        } );
    } break;
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
distribution<F>::transform_importance( const F x ) const noexcept {
    const F    xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    const auto i{ static_cast<std::uint64_t>( xval * m_M ) };
    return m_weights[i] * m_bin_count[i] * m_M
           / ( ( m_rmax - m_rmin ) * static_cast<F>( m_N ) );
}

template <std::floating_point F>
F
distribution<F>::transform_cumulative( const F x ) const noexcept {
    return m_cnorm * m_f_inverse( ( x - m_dmin ) / ( m_dmax - m_dmin ) );
}

template <std::floating_point F>
void
distribution<F>::gen_distribution(
    const distr_type type, const std::uint64_t seed, const std::uint64_t N,
    const std::uint64_t M, const F d_min, const F d_max, const F r_min,
    const F r_max, const std::function<F( const F )> & f,
    const std::function<F( const F )> & f_inverse ) noexcept {
    // Fresh initialization, allows gen_distribution to be called repeatedly
    // to generate a new, different distribution.
    initialize_distribution( type, seed, N, M, d_min, d_max, r_min, r_max, f,
                             f_inverse );

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
    case distr_type::importance: {
        // Generate flat distribution
        for ( std::size_t i = 0; i < m_N; ++i ) {
            m_bin_count[static_cast<std::uint64_t>(
                random_sample() * static_cast<F>( m_M ) )]++;
        }

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
            const F    p_flat{ m_bin_count[i] * m_M
                            / ( ( m_rmax - m_rmin ) * m_N ) };
            m_weights[i] += ( m_f( xval ) / p_flat );
        }
        // 2) For each bin in both the flat distribution and the weights,
        // divide the weight sum by the number of samples in the bin.
        for ( std::uint64_t i{ 0 }; i < m_M; ++i ) {
            m_weights[i] /= m_bin_count[i];
        }
    } break;
    case distr_type::cumulative: {
        m_cnorm = 1
                  / simpson_3_8( m_f_inverse, m_f( m_dmin ), m_f( m_dmax ),
                                 std::max<std::uint64_t>( 10 * m_N, 10000 ) );
    } break;
    default: break;
    }
    return;
}


template <std::floating_point F>
class position
{
    public:
    position( const F x, const F y, const F z ) : m_pos( { x, y, z } ) {}
    position( const std::array<F, 3> & pos ) : m_pos( pos ) {}

    [[nodiscard]] std::array<F, 3> & pos() const noexcept { return m_pos; }
    void set_pos( const std::array<F, 3> & p ) noexcept { m_pos = p; }
    [[nodiscard]] F x() const noexcept { return m_pos[0]; }
    void            set_x( const F x ) noexcept { m_pos[0] = x; }
    [[nodiscard]] F y() const noexcept { return m_pos[1]; }
    void            set_y( const F y ) noexcept { m_pos[1] = y; }
    [[nodiscard]] F z() const noexcept { return m_pos[2]; }
    void            set_z( const F z ) noexcept { m_pos[2] = z; }

    position & move( const F x, const F y, const F z ) noexcept {
        m_pos[0] += x;
        m_pos[1] += y;
        m_pos[2] += z;
        return *this;
    }
    position & move( const std::array<F, 3> & dr ) noexcept {
        m_pos[0] += dr[0];
        m_pos[1] += dr[1];
        m_pos[2] += dr[2];
        return *this;
    }

    private:
    std::array<F, 3> m_pos;
};

template <std::floating_point F>
class direction
{
    public:
    direction( const F theta, const F phi ) : m_dir( { theta, phi } ) {}
    direction( const std::array<F, 2> & dir ) : m_dir( dir ) {}

    [[nodiscard]] std::array<F, 2> & dir() const noexcept { return m_dir; }
    void set_dir( const std::array<F, 2> & d ) noexcept { m_dir = d; }
    [[nodiscard]] F theta() const noexcept { return m_dir[0]; }
    void            set_theta( const F t ) noexcept { m_dir[0] = t; }
    [[nodiscard]] F phi() const noexcept { return m_dir[1]; }
    void            set_phi( const F p ) noexcept { m_dir[1] = p; }

    private:
    std::array<F, 2> m_dir;
};


template <std::floating_point F>
class photon
{
    public:
    photon( const std::array<F, 3> & pos, const std::array<F, 2> & dir ) :
        m_pos( pos ), m_dir( dir ), m_absorbed( false ) {}

    photon & move( const F tau = 0 ) noexcept;
    photon & scatter( const std::array<F, 2> & dir ) noexcept;

    [[nodiscard]] std::array<F, 3> pos() const noexcept { return m_pos; }
    void set_pos( const std::array<F, 3> & pos ) noexcept { m_pos = pos; }

    [[nodiscard]] std::array<F, 2> dir() const noexcept { return m_dir; }
    void set_dir( const std::array<F, 2> & dir ) noexcept { m_dir = dir; }

    [[nodiscard]] F x() const noexcept { return m_pos.x(); }
    void            set_x( const F x ) { m_pos.set_x( x ); }

    [[nodiscard]] F y() const noexcept { return m_pos.y(); }
    void            set_y( const F y ) noexcept { m_pos.set_y( y ); }

    [[nodiscard]] F z() const noexcept { return m_pos.z(); }
    void            set_z( const F z ) noexcept { m_pos.set_z( z ); }

    [[nodiscard]] F theta() const noexcept { return m_dir.theta(); }
    void set_theta( const F theta ) noexcept { m_dir.set_theta( theta ); }

    [[nodiscard]] F phi() const noexcept { return m_dir.phi(); }
    void            set_phi( const F phi ) noexcept { m_dir.set_phi( phi ); }

    [[nodiscard]] F absorbed() const noexcept { return m_absorbed; }
    void            set_absorbed() noexcept { m_absorbed = true; }

    private:
    position<F>  m_pos;
    direction<F> m_dir;

    bool m_absorbed;
};

template <std::floating_point F>
photon<F> &
photon<F>::move( const F tau ) noexcept {
    const F s_theta{ std::sin( theta() ) }, c_theta{ std::cos( theta() ) },
        s_phi{ std::sin( phi() ) }, c_phi{ std::cos( phi() ) };

    set_x( x() + tau * s_theta * c_phi );
    set_y( y() + tau * s_theta * s_phi );
    set_z( z() + tau * c_theta );

    return *this;
}

template <std::floating_point F>
photon<F> &
photon<F>::scatter( const std::array<F, 2> & dir ) noexcept {
    set_theta( dir[0] );
    set_phi( dir[1] );
    return *this;
}

template <std::floating_point F>
photon<F>
track_photon( photon<F> & p, distribution<F> & d_theta, distribution<F> & d_phi,
              distribution<F> & d_albedo, distribution<F> & d_tau,
              const F tau = 10, const F albedo = 1., const F zmin = 0.,
              const F zmax = 1. ) {
    const F alpha = tau / ( zmax - zmin );
REGENERATE_PHOTON:
    p = photon<F>{ { 0, 0, 0 }, { 0, 0 } };
    goto LOOP_START;

    do {
        // Scatter or absorb?
        if ( d_albedo.random() >= albedo ) {
            // Absorb photon
            std::cout << "Photon absorbed, this is shouldn't be reached."
                      << std::endl;
            assert( false );
            p.set_absorbed();
            goto END;
        }

        // Scatter
        p.scatter( { d_theta.random(), d_phi.random() } );

    LOOP_START:

        // Move
        p.move( d_tau.random() / alpha );
    } while ( p.z() >= zmin && p.z() <= zmax );

    if ( p.z() < zmin ) {
        goto REGENERATE_PHOTON;
    }

END:
    return p;
}

template <std::floating_point F>
std::vector<photon<F>>
isotropic_scattering( const std::uint64_t N, const std::uint64_t rand_seed,
                      const F tau = 10, const F albedo = 1, const F zmin = 0.,
                      const F zmax = 0 ) {
    distribution<F> d_theta;
    distribution<F> d_phi;
    distribution<F> d_albedo;
    distribution<F> d_tau;
    // Initialize theta & phi distributions with some random seed values
    const auto d_theta_seed{ random( rand_seed, true ) };
    d_theta.gen_distribution( distr_type::flat, d_theta_seed, 2000000, 10000, 0,
                              1., 0, M_PI );
    const auto d_phi_seed{ random( rand_seed ) };
    d_phi.gen_distribution( distr_type::flat, d_phi_seed, 2000000, 10000, 0, 1.,
                            0, 2 * M_PI );
    // Albedo distribution doesn't need high N, the distribution needs to
    // vary from 0 - 1 so just use result of distribution::random_sample()
    // instead.
    const auto d_albedo_seed{ random( rand_seed ) };
    d_albedo.gen_distribution( distr_type::flat, d_albedo_seed, 2000000, 10000,
                               0, 1, 0, 1 );
    const auto d_tau_seed{ random( rand_seed ) };
    d_tau.gen_distribution(
        distr_type::cumulative, d_tau_seed, 1, 1, 0, 1, 0, 1,
        []( const F x ) { return std::exp( -x ); },
        []( const F x ) { return -std::log( 1 - x ); } );

    // Initialize N photons at position (0, 0, 0) and angle (0, 0)
    std::vector<photon<F>> photons( N, photon<F>{ { 0, 0, 0 }, { 0, 0 } } );

    // Launch particles
    std::for_each( photons.begin(), photons.end(), [&]( photon<F> & p ) {
        return track_photon<F>( p, d_theta, d_phi, d_albedo, d_tau, tau, albedo,
                                zmin, zmax );
    } );

    return photons;
}

template <std::floating_point F>
std::vector<photon<F>>
thomson_scattering() {
    return std::vector<photon<F>>{};
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

    std::cout << "1.a) Writing example distributions based on rejection method "
                 "& importance sampling to data1.csv & data2.csv. Evaluating "
                 "MSE for increasing N of both methods."
              << std::endl;

    const auto n_vals =
        std::vector<std::uint64_t>{ 1000, 10000, 100000, 1000000, 10000000 };

    std::vector<double>  error_values_rejection{};
    std::vector<double>  error_values_importance{};
    distribution<double> error_testing{};
    for ( const auto & n : n_vals ) {
        std::cout << "N: " << n << std::endl;
        const auto xvals{ linspace<double>( dmin, dmax, n ) };

        std::vector<double> true_vals( n );
        std::transform( xvals.cbegin(), xvals.cend(), true_vals.begin(),
                        []( const auto & x ) { return p_mu( x ); } );

        error_testing.gen_distribution( distr_type::rejection, 1, n, 100, dmin,
                                        dmax, rmin, rmax, p_mu<double> );
        const auto rejection_vals{ error_testing.transform( xvals ) };

        error_values_rejection.push_back( MSE( rejection_vals, true_vals ) );
        std::cout << "Rejection MSE: " << error_values_rejection.back()
                  << std::endl;

        error_testing.gen_distribution( distr_type::importance, 1, n, 100, dmin,
                                        dmax, rmin, rmax, p_mu<double> );

        const auto importance_vals{ error_testing.transform( xvals ) };

        error_values_importance.push_back( MSE( importance_vals, true_vals ) );
        std::cout << "Importance MSE: " << error_values_importance.back()
                  << std::endl;
    }

    write_to_file( "rejection_error.csv", n_vals, error_values_rejection, 20 );
    write_to_file( "importance_error.csv", n_vals, error_values_importance,
                   20 );

    distribution<double> rejection{};
    rejection.gen_distribution( distr_type::rejection,
                                static_cast<std::uint64_t>( 1 ), N, M, dmin,
                                dmax, rmin, rmax, p_mu<double> );

    const auto xvals1 = linspace<double>( dmin, dmax, M, false );
    const auto yvals1{ rejection.transform( xvals1 ) };

    write_to_file( "1a_rejection_method.csv", xvals1, yvals1, 20 );

    distribution<double> importance_sampled{};
    importance_sampled.gen_distribution( distr_type::importance,
                                         static_cast<std::uint64_t>( 1 ), N, M,
                                         dmin, dmax, rmin, rmax, p_mu<double> );

    const auto xvals2 = linspace<double>( dmin, dmax, M, false );
    const auto yvals2{ importance_sampled.transform( xvals2 ) };

    write_to_file( "1a_importance_sampled.csv", xvals2, yvals2, 20 );

    std::cout << "Done.\n" << std::endl;

    std::cout << "1.b) Simulating isotropic scattering." << std::endl;

    const std::uint64_t n_photons{ 1000000 };
    const auto          photons =
        isotropic_scattering<double>( n_photons, 121345235, 10, 1, 0., 1 );

    // Change in mu
    const std::uint64_t n{ 10 };
    const double        d_mu{ 1. / ( static_cast<double>( n ) ) };
    // Generate bin boundaries, then shift to midpoint of bin
    auto bins = linspace<double>( 0, 1, n );
    std::transform( bins.begin(), bins.end(), bins.begin(),
                    [&d_mu]( const auto & d ) { return d + ( d_mu / 2 ); } );

    // Bin intensity values
    std::vector<double> intensity( n, 0 );
    std::for_each( photons.cbegin(), photons.cend(),
                   [&intensity, &d_mu]( const auto & p ) {
                       const auto r{ std::sqrt( p.x() * p.x() + p.y() * p.y()
                                                + p.z() * p.z() ) };
                       const auto cos_theta{ p.z() / r };
                       // std::cout << "cos_theta: " << cos_theta << std::endl;
                       const auto i{ static_cast<std::uint64_t>(
                           std::cos( p.theta() ) / d_mu ) };
                       intensity[i] += 1.;
                   } );
    // Normalize by total no. of photons
    double sum{ 0. };
    std::for_each( intensity.begin(), intensity.end(),
                   [&n_photons, &sum]( auto & x ) {
                       x /= n_photons;
                       sum += x;
                   } );
    std::cout << sum << std::endl;

    for ( std::uint64_t i{ 0 }; i < intensity.size(); ++i ) {
        std::cout << i << " " << bins[i] << " " << intensity[i] << std::endl;
    }

    write_to_file( "1b_norm_intensity.csv", bins, intensity, 20 );

    std::cout << "Done.\n" << std::endl;

    std::cout << "1.c) Simulating Thomson scattering." << std::endl;
    std::cout << "Done." << std::endl;

    std::cout << std::setprecision( 20 )
              << simpson_3_8<double>(
                     []( const double x ) { return std::exp( -x ); }, 0, 10,
                     10000 )
              << std::endl;
}