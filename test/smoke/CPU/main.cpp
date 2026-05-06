#include <iostream>
#include <vector>

int main() {
	std::vector<int> test_vec = {1, 2, 3};
	if (test_vec.size() == 3) {
		std::cout << "CPU smoke test: PASS\n";
		return 0;
	}

	std::cerr << "CPU smoke test: FAIL\n";
	return 1;
}
