#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

long long
MSG_i64( const int64_t s, const int64_t a, const int64_t c, const int64_t m,
         const bool set_seed ) {
    static int64_t seed = 0;
    if ( set_seed ) {
        seed = s;
    }
    seed = ( a * seed + c ) % m;
    return seed;
}

float
MSG_f( const int64_t s, const int64_t a, const int64_t c, const int64_t m,
       const bool set_seed ) {
    static int64_t seed = 0;
    if ( set_seed ) {
        seed = s;
    }
    seed = ( a * seed + c ) % m;
    return ( float ) seed / m;
}

float
uni_binned( const int64_t seed, const int64_t a, const int64_t c,
            const int64_t m, const int64_t n_bins, const int64_t N ) {
    size_t * bins = ( size_t * ) calloc( n_bins, sizeof( size_t ) );
    float *  distr = ( float * ) calloc( N, sizeof( float ) );

    const float dx = 1 / ( float ) N;

    MSG_f( seed, a, c, m, true );
    for ( size_t i = 0; i < n_bins; ++i ) {
        distr[i] = MSG_f( seed, a, c, m, false );
        bins[( size_t ) ( distr[i] / dx )]++;
    }

    free( distr );
    free( bins );

    return 0.0;
}

int
main() {
    printf( "MSG:\n" );
    MSG_i64( 1, 16807, 0, 2147483647, true );
    for ( int i = 0; i < 10; ++i ) {
        printf( "%lld\n", MSG_i64( 1, 16807, 0, 2147483647, false ) );
    }

    putchar( '\n' );

    printf( "Uniform probability distribution:\n" );
    MSG_f( 1, 16807, 0, 2147483647, true );
    for ( int i = 0; i < 10; ++i ) {
        printf( "%.3f\n", MSG_f( 1, 16807, 0, 2147483647, false ) );
    }

    putchar( '\n' );
}

