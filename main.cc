#include "Wake_Word/main_functions.h"

int int main(int argc, char const *argv[]) {
  setup();
  while (true) {
    loop();
  }
  return 0;
}
