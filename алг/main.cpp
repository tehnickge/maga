#include <iostream>

using namespace std;

double anyFync(int a, double b) {


    return b != 0 ?  (a / b) : -1;
}

int main() {


    cout << anyFync(4,8) << endl;
    return 0;

}