#include "rdian.h"
// #include "chrono"


int main() {
    saveEngineFile("/home/user/rdian.onnx","/home/user/rdian.engine");

    RDIAN r(1920,1080);

    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        r.apply(-1);
        auto end = std::chrono::system_clock::now();
        auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "detect: " << micro << std::endl;
    }

    return -1;
}
