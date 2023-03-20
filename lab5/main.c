#include "algorithm.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Numerical integration with Simpson's 3/8 rule
float
simpson( float ( *f )( const float ), const float x1, const float x2,
         const uint64_t N ) {
    const long double dx = ( x2 - x1 ) / N;

    float sum = 0;
    for ( uint64_t i = 0; i < N; ++i ) {
        const float a = i * dx;
        const float b = ( i + 1 ) * dx;

        sum += ( ( b - a ) / 8 )
               * ( f( a ) + 3 * f( ( 2 * a + b ) / 3 )
                   + 3 * f( ( a + 2 * b ) / 3 ) + f( b ) );
    }

    return sum;
}

long long
MSG_i64( const int64_t s, const bool set_seed ) {
    static int64_t seed = 0;
    if ( set_seed ) {
        seed = s;
    }
    seed = ( 16807 * seed ) % 2147483647;
    return seed;
}

float
MSG_f( const int64_t s, const bool set_seed ) {
    static int64_t seed = 0;
    if ( set_seed ) {
        seed = s;
    }
    seed = ( 16807 * seed ) % 2147483647;
    return ( float ) seed / 2147483647;
}

typedef float ( *p_func )( const float );

int
plot( const char * const fname, const char * const restrict pre_cmds[],
      const size_t       N_cmds, const float * const restrict x,
      const float * const restrict y, const size_t N ) {
    FILE * gnu_pipe = popen( "gnuplot -persist", "w" );
    if ( !gnu_pipe ) {
        printf( "Failed to open gnuplot pipe.\n" );
        return -1;
    }

    fprintf( gnu_pipe,
             "set terminal png size 1200,900 enhanced font 'Times New "
             "Roman,14'\n" );
    fprintf( gnu_pipe, "set output '%s'\n", fname );

    if ( N_cmds != 0 && pre_cmds != NULL ) {
        for ( size_t i = 0; i < N_cmds; ++i ) {
            fprintf( gnu_pipe, "%s \n", pre_cmds[i] );
        }
    }

    if ( N != 0 && x != NULL && y != NULL ) {
        fprintf( gnu_pipe, "plot '-' notitle with linespoints linestyle 1\n" );

        for ( size_t i = 0; i < N; ++i ) {
            fprintf( gnu_pipe, "%lf %lf \n", x[i], y[i] );
        }
        fprintf( gnu_pipe, "e\n" );

        // fflush( gnu_pipe );
    }

    pclose( gnu_pipe );

    return 0;
}

typedef struct DISTRIBUTION
{
    const float      dx;
    const uint64_t * bin_count;
    const uint64_t   N;
    const uint64_t   M;
    const float      d_min;
    const float      d_max;
    const float      r_min;
    const float      r_max;
    const p_func     p_x;
} distr;

void
delete_distr( const distr * restrict d ) {
    free( ( void * ) d->bin_count );
    d = NULL;
}

// Generate Uniform Distribution
distr
GFUD_f( const int64_t s, const float dmin, const float dmax, const float rmin,
        const float rmax, const uint64_t N, const uint64_t M ) {
    // Bin width
    const float dx = ( float ) ( 1 / ( long double ) M );

    // Allocate bins
    uint64_t * bin_count = ( uint64_t * ) malloc( N * sizeof( uint64_t ) );
    if ( !bin_count ) {
        perror( "malloc" );
        exit( EXIT_FAILURE );
    }

    const float d_size = dmax - dmin;
    const float r_size = rmax - rmin;

    // Count samples per bin
    bin_count[( uint64_t ) ( MSG_f( s, true ) / dx )]++;
    for ( uint64_t i = 1; i < N; ++i ) {
        bin_count[( uint64_t ) ( MSG_f( s, false ) / dx )]++;
    }

    distr result = { .dx = dx,
                     .bin_count = bin_count,
                     .N = N,
                     .M = M,
                     .d_min = dmin,
                     .d_max = dmax,
                     .r_min = dmin,
                     .r_max = dmax };
    return result;
}

typedef enum GEN_DISTR_METHOD {
    REJECTION,
    WEIGHING,
} gen_method;

distr
rejection_method( const p_func p, const int64_t s, const float dmin,
                  const float dmax, const float rmin, const float rmax,
                  const uint64_t N, const uint64_t M ) {
    const float dx = ( float ) ( 1 / ( long double ) M );

    uint64_t * bin_count = ( uint64_t * ) malloc( N * sizeof( uint64_t ) );
    if ( !bin_count ) {
        perror( "malloc" );
        exit( EXIT_FAILURE );
    }

    const float d_size = dmax - dmin;
    const float r_size = rmax - rmin;

    float x, y;
    bool  init = true;
    do {
        x = d_size * MSG_f( s, init ) + dmin;
        y = r_size * MSG_f( s, false ) + rmin;
        init = false;
    } while ( p( x ) < y );
    bin_count[( uint64_t ) ( ( x - dmin ) / ( dx * d_size ) )]++;

    for ( uint64_t i = 1; i < N; ++i ) {
        do {
            x = d_size * MSG_f( s, false ) + dmin;
            y = r_size * MSG_f( s, false ) + rmin;
        } while ( p( x ) < y );
        bin_count[( uint64_t ) ( ( x - dmin ) / ( dx * d_size ) )]++;
    }

    printf( "TESTS:\n" );
    printf( "d_size = %lf, r_size = %lf\n", d_size, r_size );
    float    sum = 0;
    uint64_t sum2 = 0;
    for ( uint64_t i = 0; i < M; ++i ) {
        sum += ( bin_count[i] / ( dx * N ) ) * dx;
        sum2 += bin_count[i];
    }
    printf( "sum y_i * dx_i: %f\n", ( sum ) );
    printf( "sum N: %llu\n", sum2 );

    distr d = { .dx = dx,
                .bin_count = bin_count,
                .N = N,
                .M = M,
                .d_min = dmin,
                .d_max = dmax,
                .r_min = rmin,
                .r_max = rmax };

    return d;
}

distr
weighing_method( const p_func p, const int64_t s, const float dmin,
                 const float dmax, const float rmin, const float rmax,
                 const uint64_t N, const uint64_t M ) {
    const float dx = ( float ) 1 / ( long double ) M;

    uint64_t * bin_counts = ( uint64_t * ) calloc( M, sizeof( uint64_t ) );
    if ( !bin_counts ) {
        perror( "malloc" );
        exit( EXIT_FAILURE );
    }

    const float d_size = dmax - dmin;
    const float r_size = rmax - rmin;

    distr result = { 0 };
    return result;
}

distr
GUD_f( const p_func p, const gen_method method, const int64_t s,
       const float dmin, const float dmax, const float rmin, const float rmax,
       const uint64_t N, const uint64_t M ) {
    if ( !p ) {
        return GFUD_f( s, dmin, dmax, rmin, rmax, N, M );
    }

    switch ( method ) {
    case REJECTION: {
        return rejection_method( p, s, dmin, dmax, rmin, rmax, N, M );
    }
    case WEIGHING: {
        return weighing_method( p, s, dmin, dmax, rmin, rmax, N, M );
    }
    default: return GFUD_f( s, dmin, dmax, rmin, rmax, N, M );
    }
}

float
UD_f( const distr * const restrict d, const float x ) {
    const float xval = ( x - d->d_min ) / ( d->d_max - d->d_min );
    return d->bin_count[( uint64_t ) ( xval / d->dx )] * d->M
           / ( ( d->d_max - d->d_min ) * ( float ) d->N );
}

float
epsilon_UD_f( const distr * const restrict d ) {
    return sqrt( ( d->bin_count[0] - 0.1 * d->N )
                 * ( d->bin_count[0] - 0.1 * d->N )
                 / ( ( 0.1 * d->N ) * ( 0.1 * d->N ) ) );
}

// Expect peak of 2/pi
float
norm_sin_2( const float f ) {
    const float t = sin( f );
    return M_2_PI * t * t;
}

// Expect peak of 2/pi
float
norm_sin_2x( const float f ) {
    const float t = sin( 2 * f );
    return M_1_PI * t * t;
}

// Expect peak of 8/(3*pi)
float
norm_sin_4( const float f ) {
    return ( 8 / ( 3 * M_PI ) ) * pow( sin( f ), 4 );
}

float
quad( const float x ) {
    return 2 * ( x * x ) - 3 * x + 1;
}

int
main() {
    uint64_t N = 100;
    uint64_t n_bins = 10;

    /*float * x = ( float * ) malloc( 10 * sizeof( float ) );
    float * y = ( float * ) malloc( 10 * sizeof( float ) );
    if ( !x || !y ) {
        perror( "malloc" );
        exit( EXIT_FAILURE );
    }

    const uint64_t n = 8;
    for ( uint64_t i = 0; i < n; ++i ) {
        printf(
            "%llu: Plotting uniform distribution with %llu bins & %llu "
            "samples...\n",
            i, n_bins, N );
        x[i] = N;
        distr d = GFUD_f( 1, 16807, 0, 2147483647, N, n_bins );

        printf( "%f\n", UD_f( 0.5, &d ) );
        printf( "N = %llu, error = %f\n", N, epsilon_UD_f( &d ) );

        y[i] = epsilon_UD_f( &d );

        N *= 10;

        delete_distr( &d );
    }

    int64_t      n_cmds = 1;
    const char * cmds[] = { "set logscale x" };
    plot( cmds, n_cmds, x, y, n );

    free( x );
    free( y );*/

    float dmin = 0;
    float dmax = M_PI;
    float rmin = 0;
    float rmax = M_2_PI;

    N = 1000000;
    n_bins = 100;

    const distr d1 =
        GUD_f( norm_sin_2, REJECTION, 1, dmin, dmax, rmin, rmax, N, n_bins );
    printf( "%f\n", rmax );

    float * xvals = linspace( ( float ) dmin, ( float ) dmax, n_bins, false );
    float * yvals = ( float * ) malloc( n_bins * sizeof( float ) );

    for ( size_t i = 0; i < n_bins; ++i ) { yvals[i] = UD_f( &d1, xvals[i] ); }

    const char * cmds[] = { "set format '%g'", "set title '$\\sin(\\theta)$'" };
    plot( "distr1.png", cmds, 2, xvals, yvals, n_bins );

    dmax = M_PI;
    rmax = 8 / ( 3 * M_PI );
    const distr d2 =
        GUD_f( norm_sin_4, REJECTION, 1, dmin, dmax, rmin, rmax, N, n_bins );
    printf( "%f\n", rmax );

    free( xvals );

    xvals = linspace( ( float ) dmin, ( float ) dmax, n_bins, false );
    for ( size_t i = 0; i < n_bins; ++i ) { yvals[i] = UD_f( &d2, xvals[i] ); }

    cmds[0] = "set title 'distr2.'";
    plot( "distr2.png", cmds, 1, xvals, yvals, n_bins );

    free( yvals );
    delete_distr( &d1 );
}

