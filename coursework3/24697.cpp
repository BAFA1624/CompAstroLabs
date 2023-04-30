#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// Stringify things :)
#define STR( s ) #s

// Define some helper functions so arrays can be:
//  - Memberwise addition/subtraction by other arrays and floating point values.
//  - Memberwise multiplication/division by floating point values.

// -----------------------------------------------------------------------------//
// This isn't strictly necessary, I just think it's REALLY cool.
// Some template metaprogramming for efficiency
// When an array, e.g std::array<float, 10>, is +/-/etc, the compiler will
// recursively unroll the memberwise operation so that a for loop can be avoided
// which adds "jmp" calls in the generated machine code. This all happens at
// compile time, rather than at runtime. A small improvement in most cases but
// sometimes worthwhile by allowing the comiler to make further optimizations.
// -----------------------------------------------------------------------------//
template <typename T>
using trivial_op = std::function<void( T &, const T )>;

// Base cases for template metaprogramming recursive magic
template <typename T, std::size_t Size>
void
memberwise_op( [[maybe_unused]] const trivial_op<T> &       op,
               [[maybe_unused]] std::array<T, Size> &       lhs,
               [[maybe_unused]] const std::array<T, Size> & rhs,
               [[maybe_unused]] std::integral_constant<std::size_t, 0> ) {}
template <typename T, std::size_t Size>
void
memberwise_op( [[maybe_unused]] const trivial_op<T> & op,
               [[maybe_unused]] std::array<T, Size> & lhs,
               [[maybe_unused]] const T               rhs,
               [[maybe_unused]] std::integral_constant<std::size_t, 0> ) {}

template <typename T, std::size_t Size, std::size_t I>
void
memberwise_op( const trivial_op<T> & op, std::array<T, Size> & lhs,
               const std::array<T, Size> & rhs,
               std::integral_constant<std::size_t, I> ) {
    op( lhs[I - 1], rhs[I - 1] );
    memberwise_op( op, lhs, rhs, std::integral_constant<std::size_t, I - 1>() );
}
template <typename T, std::size_t Size, std::size_t I>
void
memberwise_op( const trivial_op<T> & op, std::array<T, Size> & lhs, const T rhs,
               std::integral_constant<std::size_t, I> ) {
    op( lhs[I - 1], rhs );
    memberwise_op( op, lhs, rhs, std::integral_constant<std::size_t, I - 1>() );
}

// Some macros to automatically define operator overloads for array + array,
// array + var, var * array, etc.
#define OP_FUNC( op ) []( auto & l, const auto r ) { l op r; }
#define APPLY_MEMBERWISE( op, lhs, rhs, Size )                    \
    memberwise_op( op, lhs, rhs,                                  \
                   std::integral_constant<std::size_t, Size>() ); \
    return lhs;

#define REF_OP_ARR_ARR( op )                                           \
    template <typename T, std::size_t Size>                            \
    inline std::array<T, Size> & operator op(                          \
        std::array<T, Size> & lhs, const std::array<T, Size> & rhs ) { \
        const std::function<void( T &, const T )> f = OP_FUNC( op );   \
        APPLY_MEMBERWISE( f, lhs, rhs, Size );                         \
    }
#define REF_OP_ARR_CONST( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> & operator op( std::array<T, Size> & lhs,       \
                                              const T               rhs ) {                  \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op r;                                                            \
        };                                                                     \
        memberwise_op( f, lhs, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return lhs;                                                            \
    }
#define REF_OP_CONST_ARR( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> & operator op( const T               lhs,       \
                                              std::array<T, Size> & rhs ) {    \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op r;                                                            \
        };                                                                     \
        memberwise_op( f, lhs, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return lhs;                                                            \
    }
#define REF_OP( op )       \
    REF_OP_ARR_ARR( op )   \
    REF_OP_ARR_CONST( op ) \
    REF_OP_CONST_ARR( op )

#define VAL_OP_ARR_ARR( op )                                                   \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> operator op(                                    \
        const std::array<T, Size> & lhs, const std::array<T, Size> & rhs ) {   \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op## = r;                                                        \
        };                                                                     \
        std::array<T, Size> tmp{ lhs };                                        \
        memberwise_op( f, tmp, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return tmp;                                                            \
    }
#define VAL_OP_ARR_CONST( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> operator op( const std::array<T, Size> & lhs,   \
                                            const T                     rhs ) {                    \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op## = r;                                                        \
        };                                                                     \
        std::array<T, Size> tmp{ lhs };                                        \
        memberwise_op( f, tmp, rhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return tmp;                                                            \
    }
#define VAL_OP_CONST_ARR( op )                                                 \
    template <typename T, std::size_t Size>                                    \
    inline std::array<T, Size> operator op(                                    \
        const T lhs, const std::array<T, Size> & rhs ) {                       \
        const std::function<void( T &, const T )> f = []( T & l, const T r ) { \
            l op## = r;                                                        \
        };                                                                     \
        std::array<T, Size> tmp{ rhs };                                        \
        memberwise_op( f, tmp, lhs,                                            \
                       std::integral_constant<std::size_t, Size>() );          \
        return tmp;                                                            \
    }
#define VAL_OP( op )       \
    VAL_OP_ARR_ARR( op )   \
    VAL_OP_ARR_CONST( op ) \
    VAL_OP_CONST_ARR( op )

// Required std::array operators
// clang-format off
REF_OP( += )
REF_OP( -= )
REF_OP( *= )
REF_OP( /= )
VAL_OP( + )
VAL_OP( - )
VAL_OP( * )
VAL_OP( / )
// clang-format on

// Simple std::array printer
template <typename T, std::size_t Size>
void
print_array( const std::array<T, Size> & a ) {
    std::string s{ "array: { " };
    for ( std::uint64_t i{ 0 }; i < Size; ++i ) {
        s += std::to_string( a[i] ) + ", ";
    }
    s.pop_back();
    s.pop_back();
    s += " }";
    std::cout << s << std::endl;
}

// Writes a set of arrays to file
template <typename T, std::size_t N>
void
write_to_file( const std::string &                   filename,
               const std::vector<std::array<T, N>> & values,
               const std::vector<std::string> &      keys = {} ) {
    assert( keys.size() == values.size() || keys.size() == 0 );

    // Find shortest array length, limits the number of rows written to file
    std::uint64_t n_rows{ std::numeric_limits<std::uint64_t>::max() };
    for ( const auto & v : values ) {
        n_rows = std::min<std::uint64_t>( n_rows, v.size() );
    }

    // Open file & write column titles if provided
    std::ofstream fp( filename );
    if ( keys.size() > 0 ) {
        std::string header{ "" };
        for ( const auto & key : keys ) { header += key + ","; }
        header.pop_back();
        header += "\n";
        fp << header.c_str();
    }

    // Write data
    for ( std::uint64_t i{ 0 }; i < n_rows; ++i ) {
        std::string line{ "" };
        for ( const auto & v : values ) {
            line += std::to_string( v[i] ) + ",";
        }
        line.pop_back();
        line += "\n";
        fp << line.c_str();
    }
}

// Define "state" & "flux" as fixed size (default = 3) arrays.
template <typename T, std::size_t Size = 3>
using state = std::array<T, Size>;
template <typename T, std::size_t Size = 3>
using flux = state<T, Size>;

template <typename T, std::size_t Size>
constexpr std::array<state<T>, Size>
construct_state( const std::array<T, Size> & q1, const std::array<T, Size> & q2,
                 const std::array<T, Size> & q3 ) {
    std::array<state<T>, Size> state_array;

    for ( std::size_t i{ 0 }; i < Size; ++i ) {
        state_array[i] = state<T>{ q1[i], q2[i], q3[i] };
    }

    return state_array;
}

// Class enum for selecting between type of algorithm.
// The value of each enum name is set to the required no. of ghost cells
enum class solution_type : std::size_t { lax_friedrichs };
enum class boundary_type : std::size_t { outflow, reflecting, custom };

// Fluid dynamics solver definition
template <typename T, std::size_t Size, solution_type Type, boundary_type Lbc,
          boundary_type Rbc>
class fluid_solver
{
    public:
    fluid_solver( const std::array<state<T>, Size> & initial_state ) {
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + 1] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    fluid_solver( const std::array<T, Size> & q1,
                  const std::array<T, Size> & q2,
                  const std::array<T, Size> & q3 ) {
        const auto initial_state{ construct_state( q1, q2, q3 ) };
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + 1] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }

    void initialize_state(
        const std::array<state<T>, Size> & initial_state ) noexcept {
        for ( std::size_t i{ 0 }; i < Size; ++i ) {
            m_state[i + 1] = initial_state[i];
        }
        m_previous_state = m_state;
        apply_boundary_conditions();
    }
    void intialize_state( const std::array<T, Size> & q1,
                          const std::array<T, Size> & q2,
                          const std::array<T, Size> & q3 ) {
        const auto initial_state{ construct_state( q1, q2, q3 ) };
        initialize_state( initial_state );
    }

    [[nodiscard]] constexpr auto & state() const noexcept { return m_state; }
    [[nodiscard]] constexpr auto & previous_state() const noexcept {
        return m_previous_state;
    }
    [[nodiscard]] constexpr auto q1() const noexcept {
        std::array<T, Size> q1_array;
        for ( std::size_t i{ 0 }; i < Size ) {
            q1_array[i] = m_state[i + 1][0];
        }
        return q1_array;
    }
    [[nodiscard]] constexpr auto q2() const noexcept {
        std::array<T, Size> q2_array;
        for ( std::size_t i{ 0 }; i < Size ) {
            q2_array[i] = m_state[i + 1][1];
        }
        return q2_array;
    }
    [[nodiscard]] constexpr auto q3() const noexcept {
        std::array<T, Size> q2_array;
        for ( std::size_t i{ 0 }; i < Size ) {
            q3_array[i] = m_state[i + 1][2];
        }
        return q3_array;
    }

    constexpr auto simulate( const T time_step, const T endpoint,
                             const std::size_t n_saves = 0 ) noexcept {
        switch ( Type ) {
        case solution_type::lax_friedrichs: {
        } break;
        }
        return m_state;
    }

    private:
    constexpr void apply_boundary_conditions() noexcept {
        switch ( Lbc ) {
        case boundary_type::outflow: {
            m_state[0] = m_state[1];
            m_previous_state[0] = m_previous_state[1];
        } break;
        case boundary_type::reflecting: {
            m_state[0] = m_state[1];
            m_previous_state[0] = m_previous_state[1];
            m_state[0][1] *= -1;
            m_previous_state[0][1] *= -1;
        } break;
        }
        switch ( Rbc ) {
        case boundary_type::outflow: {
            m_state[Size + 1] = m_state[Size];
            m_previous_state[Size + 1] = m_previous_state[Size];
        } break;
        case boundary_type::reflecting: {
            m_state[Size + 1] = m_state[Size];
            m_previous_state[Size + 1] = m_previous_state[Size];
            m_state[Size + 1][1] *= -1;
            m_previous_state[Size + 1][1] *= -1;
        } break;
        }
    };

    // An array of state arrays (3 values) with length Size + 2 * no. of
    // ghost cells each side
    std::array<state<T>, Size + 2> m_state;
    std::array<state<T>, Size + 2> m_previous_state;
};

int
main() {
    std::array<double, 5> a1;
    a1.fill( 1.0 );
    std::array<double, 5> a2;
    a2.fill( 2.0 );
    std::array<double, 5> a3;
    a3.fill( 3.0 );

    const auto initial_state = construct_state( a1, a2, a3 );
    /*for ( std::size_t i{ 0 }; i < initial_state.size(); ++i ) {
        print_array( initial_state[i] );
    }*/

    fluid_solver<double, std::tuple_size_v<decltype( initial_state )>,
                 solution_type::lax_friedrichs, boundary_type::outflow,
                 boundary_type::reflecting>
        fs( initial_state );
}