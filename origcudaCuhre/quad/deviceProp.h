
namespace quad{
  class CUDADevice{
  public:
    // Version information
    static int     sm_version;                // SM version of target device (SM version X.YZ in XYZ integer form)
    static const int     ptx_version = 0;            // Bundled PTX version for target device (PTX version X.YZ in XYZ integer form)
    
  };
}
