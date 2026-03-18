#include <cstdint>

#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
const string BatchSizeResourceName = "BatchSizeResource_";
class BatchSizeResource : public ResourceBase {
  public:
    ~BatchSizeResource() override {
      VLOG(1) << "BatchSizeResource destroyed with batch size: " << batch_size_;
    }
    string DebugString() const override { return BatchSizeResourceName; }
    void SetBatchSize(int64_t s) { batch_size_ = s; }
    int64_t GetBatchSize() const { return batch_size_; }
  private:
    int64_t batch_size_ = 0;
};
}
