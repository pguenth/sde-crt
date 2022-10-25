#include <iostream>
#include "batch/pseudoparticlestate.h"
#include "batch/breakpointstate.h"
#include <vector>
#include <string>
#include <chrono>

typedef std::chrono::high_resolution_clock::time_point time_point;
typedef std::chrono::high_resolution_clock::duration duration;
inline time_point now() { return std::chrono::high_resolution_clock::now(); }
inline double duration_ms(duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main(){
    std::cout << "Test\n";

    /*
    time_point start_time = now();
    BatchSourcetest batch{0, 10000, 1, -1, 1};
    batch.run();

    std::cout << duration_ms(now() - start_time) << '\n';
    std::vector<PseudoParticleState> results = batch.states();
    for (auto &s : results){
    //    std::cout << (std::string)s << '\n';
    }
    */

    return 0;
}

