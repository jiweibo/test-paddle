#include "engine.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/include/paddle_inference_api.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<std::shared_ptr<paddle_infer::Predictor>> predictors1;
std::vector<std::shared_ptr<paddle_infer::Predictor>> predictors2;
int reload_num = 0;

auto predictors = &predictors1;

void initPredictors(const char *model_dir, const char *model_file,
                    const char *params_file, const int32_t thread_num,
                    const int32_t math_thread, const bool use_lite) {
  std::vector<std::shared_ptr<paddle_infer::Predictor>> *p = nullptr;
  if (reload_num % 2 == 0) {
    p = &predictors1;
  } else {
    p = &predictors2;
  }

  p->resize(thread_num);

  paddle_infer::Config config;
  if (strlen(model_dir) == 0) {
    // Load combined model.
    config.SetModel(model_file, params_file);
  } else {
    // Load uncombined model.
    config.SetModel(model_dir);
  }

  config.DisableGlogInfo();
  config.SetCpuMathLibraryNumThreads(math_thread);

  if (use_lite) {
    // config.EnableLiteEngine(paddle_infer::PrecisionType::kFloat32, false);
  }

  auto main_predictor = paddle_infer::CreatePredictor(config);

  for (int i = 1; i < thread_num; ++i) {
    (*p)[i] = std::move(main_predictor->Clone());
  }
  (*p)[0] = std::move(main_predictor);
}

void InitPredictor(const char *model_dir, const char *model_file,
                   const char *params_file, const int32_t thread_num,
                   const int32_t math_thread, const bool use_lite) {
  initPredictors(model_dir, model_file, params_file, thread_num, math_thread,
                 use_lite);
}

void SetInt64Data(int reload_id, int key_id, const char *name,
                  const int64_t *data, const int32_t *shape, int shape_size) {
  std::vector<std::shared_ptr<paddle_infer::Predictor>> *pr = nullptr;
  if (reload_id == 0) {
    pr = &predictors1;
  } else {
    pr = &predictors2;
  }

  auto &p = (*pr)[key_id];
  auto handle = p->GetInputHandle(name);
  std::vector<int> shape_vec(shape, shape + shape_size);
  handle->Reshape(shape_vec);
  handle->CopyFromCpu(data);
}

void SetFloat32Data(int reload_id, int key_id, const char *name,
                    const float *data, const int *shape, int shape_size) {
  std::vector<std::shared_ptr<paddle_infer::Predictor>> *pr = nullptr;
  if (reload_id == 0) {
    pr = &predictors1;
  } else {
    pr = &predictors2;
  }
  auto &p = (*pr)[key_id];

  auto handle = p->GetInputHandle(name);
  std::vector<int> shape_vec(shape, shape + shape_size);
  handle->Reshape(shape_vec);
  handle->CopyFromCpu(data);
}

void Run(int reload_id, int key_id, float *res0, float *res1, float *res2) {
  std::vector<std::shared_ptr<paddle_infer::Predictor>> *pr = nullptr;
  if (reload_id == 0) {
    pr = &predictors1;
  } else {
    pr = &predictors2;
  }
  auto &p = (*pr)[key_id];

  p->Run();
  auto out_names = p->GetOutputNames();
  auto out_handle0 = p->GetOutputHandle(out_names[0]);
  auto out_handle1 = p->GetOutputHandle(out_names[1]);
  auto out_handle2 = p->GetOutputHandle(out_names[2]);
  out_handle0->CopyToCpu(res0);
  out_handle1->CopyToCpu(res1);
  out_handle2->CopyToCpu(res2);
}

// data1: type is float, len is batch_size * 1 * 172, data is [field_0, field_1,
// ...] data2 int64, [batch_size, 1] x (172-344) data2: type is int64, len is
// batch_size * (345 - 172) + batch_size * 10 * (358 - 345), data is [field_172,
// field_173, ..., field_357]
void RunNormal(int key_id, int batch_size, float *data1, int64_t *data2,
               float *res0, float *res1, float *res2) {
  auto &p = (*predictors)[key_id];
  auto out_names = p->GetOutputNames();

  for (int i = 0; i < 172; ++i) {
    auto name = "field_" + std::to_string(i);
    auto in_handle = p->GetInputHandle(name);
    in_handle->Reshape({batch_size, 1});
    in_handle->CopyFromCpu(data1 + batch_size * i);
  }

  for (int i = 172; i < 344; ++i) {
    auto name = "field_" + std::to_string(i);
    auto in_handle = p->GetInputHandle(name);
    in_handle->Reshape({batch_size, 1});
    in_handle->CopyFromCpu(data2 + batch_size * (i - 172));
  }

  int64_t *data3 = data2 + batch_size * (344 - 172);
  int magic_num = 10;

  for (int i = 344; i < 358; ++i) {
    auto name = "field_" + std::to_string(i);
    auto in_handle = p->GetInputHandle(name);
    in_handle->Reshape({batch_size, 10});
    in_handle->CopyFromCpu(data3 + (i - 344) * batch_size * magic_num);
  }

  p->Run();

  auto out_handle0 = p->GetOutputHandle(out_names[0]);
  auto out_handle1 = p->GetOutputHandle(out_names[1]);
  auto out_handle2 = p->GetOutputHandle(out_names[2]);
  out_handle0->CopyToCpu(res0);
  out_handle1->CopyToCpu(res1);
  out_handle2->CopyToCpu(res2);
}

void Run160Model(int key_id, int batch_size, int64_t *data, float *res0,
                 float *res1, float *res2) {
  auto &p = (*predictors)[key_id];
  auto out_names = p->GetOutputNames();

  for (int i = 0; i < 160; ++i) {
    auto name = "field_" + std::to_string(i);
    auto in_handle = p->GetInputHandle(name);
    in_handle->Reshape({batch_size, 1});
    in_handle->CopyFromCpu(data + batch_size * i);
  }

  p->Run();

  auto out_handle0 = p->GetOutputHandle(out_names[0]);
  auto out_handle1 = p->GetOutputHandle(out_names[1]);
  auto out_handle2 = p->GetOutputHandle(out_names[2]);
  out_handle0->CopyToCpu(res0);
  out_handle1->CopyToCpu(res1);
  out_handle2->CopyToCpu(res2);
}

int Reload(const char *model_dir, const char *model_file,
           const char *params_file, const int32_t thread_num,
           const int32_t math_thread, const bool use_lite, int model_id) {
  reload_num++;

  // init new predictor.
  InitPredictor(model_dir, model_file, params_file, thread_num, math_thread,
                use_lite);

  // we should also reload the old predictor, but they may be run, so we don't
  // do anything because of std::shared_ptr.
  if (reload_num % 2 == 1) {
    predictors = &predictors2;
    return 1;
  } else {
    predictors = &predictors1;
    return 0;
  }
}
