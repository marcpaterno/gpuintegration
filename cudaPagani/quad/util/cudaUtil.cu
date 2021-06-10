#include "cudaUtil.h"

CommandLineArgs::CommandLineArgs(int argc, char** argv)
{
  using namespace std;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if ((arg[0] != '-') || (arg[1] != '-')) {
      continue;
    }
    string::size_type pos;
    string key, val;
    if ((pos = arg.find('=')) == string::npos) {
      key = string(arg, 2, arg.length() - 2);
      val = "";
    } else {
      key = string(arg, 2, pos - 2);
      val = string(arg, pos + 1, arg.length() - 1);
    }
    pairs[key] = val;
  }
}

bool
CommandLineArgs::CheckCmdLineFlag(const char* arg_name) const
{
  auto const itr = pairs.find(arg_name);
  return itr != pairs.end();
}
