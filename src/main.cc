// Unit test main. Test model inferer and converter here.
// Do nothing.

#include <gtest/gtest.h>

#include <memory>

TEST(LocalTest, TestInit) {
}

TEST(LocalTest, TestInfer) {}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
