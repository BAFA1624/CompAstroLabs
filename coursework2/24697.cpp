#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
<<<<<<< HEAD
<<<<<<< HEAD
=======
#include <concepts>
>>>>>>> coursework3_array
=======
#include <concepts>
=======
<<<<<<< HEAD
=======
#include <concepts>
>>>>>>> coursework3_array
>>>>>>> hand-in
>>>>>>> 39ae558f7f62c32f1ca7af80926efdfff52340ec
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>


template <typename T>
std::vector<T>
linspace( const T & start, const T & end, const std::uint64_t N,
          const bool include_endpoint = false ) {
    const T dx{
        ( start < end ) ?
            ( end - start ) / static_cast<T>( include_endpoint ? N - 1 : N ) :
            -( start - end ) / static_cast<T>( include_endpoint ? N - 1 : N )
    };

    std::vector<T> result( N, 0 );
    T              tmp{ start };
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        result[i] = tmp;
        tmp += dx;
    }

    return result;
}

template <typename T>
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

template <typename T1, typename T2>
void
write_to_file( const std::string & fname, const std::vector<T1> & x,
               const std::vector<T2> & y, const std::uint64_t precision ) {
    const std::uint64_t n_rows{ std::min( x.size(), y.size() ) };
    std::ofstream       f( fname, std::ios_base::out );
    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        f << std::setprecision( precision ) << x[i] << "," << y[i] << "\n";
    }
}


enum class distr_type { flat, rejection, importance, cumulative };

template <typename F>
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
        switch ( m_type ) {
        case distr_type::rejection: {
            F rand_x{ ( m_dmax - m_dmin ) * random_sample() + m_dmin };
            F rand_y{ ( m_rmax - m_rmin ) * random_sample() + m_rmin };
            while ( rand_y < m_f( rand_x ) ) {
                rand_x = ( m_dmax - m_dmin ) * random_sample() + m_dmin;
                rand_y = ( m_rmax - m_rmin ) * random_sample() + m_rmin;
            }
            return rand_x;
        } break;
        case distr_type::cumulative: {
            return m_f_inverse( random_sample() );
        } break;
        default: {
            return ( m_rmax - m_rmin ) * random_sample() + m_rmin;
        } break;
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

template <typename F>
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

template <typename F>
F
distribution<F>::transform( const F x ) const noexcept {
    return ( m_type == distr_type::importance ) ? transform_importance( x ) :
                                                  transform_nonweighted( x );
}

template <typename F>
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

template <typename F>
F
distribution<F>::transform_nonweighted( const F x ) const noexcept {
    const F xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    return ( ( m_bin_count[static_cast<std::uint64_t>(
                   xval * static_cast<F>( m_M ) )]
               * m_M / ( ( m_dmax - m_dmin ) * static_cast<F>( m_N ) ) )
             + m_rmin );
}

template <typename F>
F
distribution<F>::transform_importance( const F x ) const noexcept {
    const F    xval{ ( x - m_dmin ) / ( m_dmax - m_dmin ) };
    const auto i{ static_cast<std::uint64_t>( xval * m_M ) };
    return m_weights[i] * m_bin_count[i] * m_M
           / ( ( m_rmax - m_rmin ) * static_cast<F>( m_N ) );
}

template <typename F>
F
distribution<F>::transform_cumulative( const F x ) const noexcept {
    return m_f_inverse( ( x - m_dmin ) / ( m_dmax - m_dmin ) );
}

template <typename F>
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
    default: break;
    }
    return;
}


enum class scattering_type { isotropic, thomson };

template <typename F>
class photon
{
    public:
    photon( const F x, const F y, const F z, const F theta, const F phi ) :
        m_x( x ),
        m_y( y ),
        m_z( z ),
        m_theta( theta ),
        m_phi( phi ),
        m_absorbed( false ) {}

    photon & move( const F tau = 0 ) noexcept;
    photon & scatter( const F theta, const F phi ) noexcept;
    photon & set( const F x, const F y, const F z, const F theta, const F phi,
                  const bool absorbed ) noexcept {
        m_x = x;
        m_y = y;
        m_z = z;
        m_theta = theta;
        m_phi = phi;
        m_absorbed = absorbed;
        return *this;
    }

    [[nodiscard]] std::tuple<F, F, F> pos() const noexcept {
        return { m_x, m_y, m_z };
    }

    [[nodiscard]] std::tuple<F, F> dir() const noexcept {
        return { m_theta, m_phi };
    }

    [[nodiscard]] F x() const noexcept { return m_x; }
    void            set_x( const F x ) noexcept { m_x = x; }

    [[nodiscard]] F y() const noexcept { return m_y; }
    void            set_y( const F y ) noexcept { m_y = y; }

    [[nodiscard]] F z() const noexcept { return m_z; }
    void            set_z( const F z ) noexcept { m_z = z; }

    [[nodiscard]] F theta() const noexcept { return m_theta; }
    void            set_theta( const F theta ) noexcept { m_theta = theta; }

    [[nodiscard]] F phi() const noexcept { return m_phi; }
    void            set_phi( const F phi ) noexcept { m_phi = phi; }

    [[nodiscard]] F absorbed() const noexcept { return m_absorbed; }
    void            set_absorbed( const bool b ) noexcept { m_absorbed = b; }

    private:
    F m_x;
    F m_y;
    F m_z;

    F m_theta;
    F m_phi;

    bool m_absorbed;
};

template <typename F>
photon<F> &
photon<F>::move( const F ds ) noexcept {
    const F s_theta{ std::sin( m_theta ) }, c_theta{ std::cos( m_theta ) },
        s_phi{ std::sin( m_phi ) }, c_phi{ std::cos( m_phi ) };

    m_x += ds * s_theta * c_phi;
    m_y += ds * s_theta * s_phi;
    m_z += ds * c_theta;

    return *this;
}

template <typename F>
photon<F> &
photon<F>::scatter( const F theta, const F phi ) noexcept {
    m_theta = theta;
    m_phi = phi;
    return *this;
}

template <typename F>
photon<F> &
track_photon_isotropic( photon<F> & p, distribution<F> & d_theta,
                        distribution<F> & d_phi, distribution<F> & d_albedo,
                        distribution<F> & d_tau, const F tau = 10,
                        const F albedo = 1., const F zmin = 0.,
                        const F zmax = 1. ) {
    const F alpha{ tau / ( zmax - zmin ) };

    while ( p.z() <= zmax ) {
        if ( p.z() < zmin ) {
            p.set( 0, 0, 0, 0, 0, false );
        }

        p.move( d_tau.random() / alpha );

        if ( d_albedo.random() >= albedo ) {
            std::cout << "Photon absorbed, this shouldn't be reached atm."
                      << std::endl;
            p.set_absorbed( true );
            return p;
        }

        p.scatter( d_theta.random(), d_phi.random() );
    }
    return p;
}

int p_count = 0;
template <typename F>
photon<F>
track_photon_thomson( photon<F> & p, distribution<F> & d_theta,
                      distribution<F> & d_phi, distribution<F> & d_albedo,
                      distribution<F> & d_tau, const F tau = 10,
                      const F albedo = 1., const F zmin = 0.,
                      const F zmax = 1. ) {
    const F alpha{ tau / ( zmax - zmin ) };
    p_count++;

    // First move straight upwards. Lab & photon frames aligned.
    p.move( d_tau.random() / alpha );

    while ( p.z() < zmax ) {
        if ( p.z() < zmin ) {
            p.set( 0, 0, 0, 0, 0, false );
        }

        // Check for scattering
        if ( d_albedo.random() >= albedo ) {
            std::cout << "Photon absorbed, this shouldn't happen." << std::endl;
            p.set_absorbed( true );
            return p;
        }

        // Get scattering angles in photon's frame of reference
        const F theta_photon{ d_theta.random() }, phi_photon{ d_phi.random() };
        const F sin_theta_photon{ std::sin( theta_photon ) },
            cos_theta_photon{ std::cos( theta_photon ) },
            sin_phi_photon{ std::sin( phi_photon ) },
            cos_phi_photon{ std::cos( phi_photon ) },
            sin_theta_lab{ std::sin( p.theta() ) },
            cos_theta_lab{ std::cos( p.theta() ) },
            sin_phi_lab{ std::sin( p.phi() ) },
            cos_phi_lab{ std::cos( p.phi() ) };

        // Magnitude of distance moved in photon's frame.
        const F dr_photon{ d_tau.random() / alpha };

        // Calculate distances moved in photon's frame
        const F dx_photon{ dr_photon * sin_theta_photon * cos_phi_photon };
        const F dy_photon{ dr_photon * sin_theta_photon * sin_phi_photon };
        const F dz_photon{ dr_photon * cos_theta_photon };

        // Assume photon's x-axis lies in the x-y plane of the lab frame
        // Align y-axis
        const F dx_align_y{ dx_photon * cos_phi_lab - dy_photon * sin_phi_lab };
        const F dy_align_y{ dx_photon * sin_phi_lab + dy_photon * cos_phi_lab };

        // Transfer back to lab frame
        const F dx{ dx_align_y * cos_theta_lab + dz_photon * sin_theta_lab };
        const F dz{ dz_photon * cos_theta_lab - dx_align_y * sin_theta_lab };

        p.set_x( p.x() + dx );
        p.set_y( p.y() + dy_align_y );
        p.set_z( p.z() + dz );

        const F r{ p.x() * p.x() + p.y() * p.y() + p.z() * p.z() };
        p.set_theta( std::acos( p.z() * p.z() / r ) );
        p.set_phi( std::atan2( p.y(), p.x() ) );
    }

    return p;
}

template <typename F>
std::vector<photon<F>>
launch_simulation( const scattering_type type, const std::uint64_t N,
                   const std::uint64_t rand_seed, const F tau = 10,
                   const F albedo = 1, const F zmin = 0., const F zmax = 0 ) {
    distribution<F> d_theta;
    distribution<F> d_phi;
    distribution<F> d_albedo;
    distribution<F> d_tau;

    const auto d_theta_seed{ random( rand_seed, true ) };
    const auto d_phi_seed{ random( rand_seed ) };
    const auto d_albedo_seed{ random( rand_seed ) };
    const auto d_tau_seed{ random( rand_seed ) };

    // Initialize N photons at position (0, 0, 0) and angle (0, 0)
    std::vector<photon<F>> photons( N, photon<F>{ 0, 0, 0, 0, 0 } );

    // Launch particles
    switch ( type ) {
    case scattering_type::isotropic: {
        // Initialize theta & phi distributions with some random seed values
        d_theta.gen_distribution( distr_type::flat, d_theta_seed, 1000000,
                                  10000, 0, 1., 0, M_PI );
        d_phi.gen_distribution( distr_type::flat, d_phi_seed, 1000000, 10000, 0,
                                1., 0, 2 * M_PI );
        // Albedo distribution doesn't need high N, the distribution needs to
        // vary from 0 - 1 so just use result of distribution::random_sample()
        // instead.
        d_albedo.gen_distribution( distr_type::flat, d_albedo_seed, 1000000,
                                   10000, 0, 1, 0, 1 );
        d_tau.gen_distribution(
            distr_type::cumulative, d_tau_seed, 1, 1, 0, 1, 0, 1,
            []( const F x ) { return std::exp( -x ); },
            []( const F y ) { return -std::log( 1 - y ); } );
        std::for_each( photons.begin(), photons.end(), [&]( photon<F> & p ) {
            return track_photon_isotropic<F>( p, d_theta, d_phi, d_albedo,
                                              d_tau, tau, albedo, zmin, zmax );
        } );
    } break;
    case scattering_type::thomson: {
        d_theta.gen_distribution( distr_type::importance, d_theta_seed, 1000000,
                                  10000, 0., 1., 0, M_PI, []( const double x ) {
                                      const F mu{ std::cos( x ) };
                                      return 0.375 * ( 1 + mu * mu );
                                  } );
        d_phi.gen_distribution( distr_type::flat, d_phi_seed, 1000000, 10000,
                                0., 1., 0, M_PI );
        d_albedo.gen_distribution( distr_type::flat, d_albedo_seed, 1000000,
                                   10000, 0, 1, 0, 1 );
        d_tau.gen_distribution(
            distr_type::cumulative, d_tau_seed, 1, 1, 0, 1, 0, 1,
            []( const F x ) { return std::exp( -x ); },
            []( const F y ) { return -std::log( 1 - y ); } );

        std::for_each( photons.begin(), photons.end(), [&]( photon<F> & p ) {
            return track_photon_thomson<F>( p, d_theta, d_phi, d_albedo, d_tau,
                                            tau, albedo, zmin, zmax );
        } );
    } break;
    default: break;
    };

    return photons;
}

template <typename F>
std::tuple<std::vector<F>, std::vector<F>>
norm_intensity( const std::vector<photon<F>> & photons,
                const std::uint64_t            N ) {
    // Create bins
    const F        d_mu{ 1 / static_cast<F>( N ) };
    std::vector<F> bins( N );
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        bins[i] = ( i * d_mu ) + ( d_mu / 2 );
    }

    std::vector<F> intensity( N, 0. );
    F              sum{ 0. };
    for ( const auto & p : photons ) {
        const F r{ std::sqrt( p.x() * p.x() + p.y() * p.y() + p.z() * p.z() ) };

        // Exception for when photon leaves directly upwards
        // in first step.
        std::uint64_t i{ 0 };
        if ( std::fabs( p.z() ) != r ) {
            i = static_cast<std::uint64_t>( std::fabs( p.z() ) / ( r * d_mu ) );
        }
        else {
            i = N - 1;
        }
        sum += 1.;
        intensity[i]++;
    }

    for ( auto & i : intensity ) { i /= photons.size(); }

    return { bins, intensity };
}

template <typename F>
F
p_mu( const F m ) {
    return 0.375 * ( 1 + pow( m, 2.0 ) );
}

int
main() {
    // Part 1.a
    const std::uint64_t N{ 1000000 };
    const std::uint64_t M{ 1000 };

    const double dmin{ -1 }, dmax{ 1 }, rmin{ 0. }, rmax{ 0.75 };

    std::cout << "1.a) Writing example distributions based on rejection "
                 "method "
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



    // Part 1.b
    std::cout << "1.b) Simulating isotropic scattering." << std::endl;
    const std::uint64_t seed{ 1 };
    const double optical_depth{ 10 }, albedo{ 1. }, zmin{ 0. }, zmax{ 1. };

    const std::uint64_t n_photons{ 1000000 };
    const auto          isotropic_photons =
        launch_simulation<double>( scattering_type::isotropic, n_photons, seed,
                                   optical_depth, albedo, zmin, zmax );

    const auto & [isotropic_bins, isotropic_intensity] =
        norm_intensity( isotropic_photons, 10 );

    write_to_file( "1b_norm_intensity_tau_" + std::to_string( optical_depth )
                       + ".csv",
                   isotropic_bins, isotropic_intensity, 20 );

    std::cout << "Done.\n" << std::endl;


    // Part 1.c
    std::cout << "1.c) Simulating Thomson scattering." << std::endl;

    const auto thomson_photons =
        launch_simulation<double>( scattering_type::thomson, n_photons, seed,
                                   optical_depth, albedo, zmin, zmax );

    const auto & [thomson_bins, thomson_intensity] =
        norm_intensity( thomson_photons, 10 );

    write_to_file( "1c_norm_intensity_tau_" + std::to_string( optical_depth )
                       + ".csv",
                   thomson_bins, thomson_intensity, 20 );
    std::cout << "Done." << std::endl;
}
