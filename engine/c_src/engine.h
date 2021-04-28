#include <stdbool.h>
#include <stdint.h>

#define CAPI_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

CAPI_EXPORT void InitPredictor(const char *model_dir, const char *model_file,
                               const char *params_file,
                               const int32_t thread_num,
                               const int32_t math_thread, const bool use_lite);

CAPI_EXPORT void SetInt64Data(int reload_id, int key_id, const char *name,
                              const int64_t *data, const int *shape,
                              int shape_size);

CAPI_EXPORT void SetFloat32Data(int reload_id, int key_id, const char *name,
                                const float *data, const int *shape,
                                int shape_size);

CAPI_EXPORT void Run(int reload_id, int key_id, float *res0, float *res1,
                     float *res2);

CAPI_EXPORT void RunNormal(int key_id, int batch_size, float *data1,
                           int64_t *data2, float *res0, float *res1,
                           float *res2);

CAPI_EXPORT void Run160Model(int key_id, int batch_size, int64_t *data,
                             float *res0, float *res1, float *res2);

CAPI_EXPORT int Reload(const char *model_dir, const char *model_file,
                       const char *params_file, const int32_t thread_num,
                       const int32_t math_thread, const bool use_lite,
                       int model_id);

// todo: 多模型加载，可以trick，c++写多个接口

#ifdef __cplusplus
}
#endif
