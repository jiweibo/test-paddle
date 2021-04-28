package engine

// #cgo CFLAGS: -I${SRCDIR}/ -I${SRCDIR}/paddle_inference/
// #cgo LDFLAGS: -L${SRCDIR}/ -L${SRCDIR}/paddle_inference/paddle/lib -L${SRCDIR}/paddle_inference/third_party/install/lite/cxx/lib -lengine -lpaddle_inference -lpaddle_full_api_shared
// #include "engine.h"
// #include <stdlib.h>
// #include <stdbool.h>
import "C"
import (
	"unsafe"
)

func InitEngine(model_dir string, model_file string, params_file string, thread_num int, math_num int, use_lite bool) {
	c_model_file := C.CString(model_file)
	c_params_file := C.CString(params_file)
	c_model_dir := C.CString(model_dir)

	C.InitPredictor(c_model_dir, c_model_file, c_params_file, C.int(thread_num), C.int(math_num), C.bool(use_lite))
	defer func() {
		C.free(unsafe.Pointer(c_model_dir))
		C.free(unsafe.Pointer(c_model_file))
		C.free(unsafe.Pointer(c_params_file))
	}()
}

func SetInputData(reload_id int32, key_id int, name string, data interface{}, shape []int32) {
	c_name := C.CString(name)
	switch v := data.(type) {
	case []int64:
		C.SetInt64Data(C.int32_t(reload_id), C.int(key_id), c_name, (*C.int64_t)(unsafe.Pointer(&v[0])), (*C.int32_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)))
	case []float32:
		C.SetFloat32Data(C.int32_t(reload_id), C.int(key_id), c_name, (*C.float)(unsafe.Pointer(&v[0])), (*C.int32_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)))
	default:
		panic("not supported input type. only support []float32 and []int64")
	}

	defer func() {
		C.free(unsafe.Pointer(c_name))
	}()
}

func Run(reload_id int32, key_id int, res0, res1, res2 []float32) {
	C.Run(C.int32_t(reload_id), C.int(key_id), (*C.float)(unsafe.Pointer(&res0[0])), (*C.float)(unsafe.Pointer(&res1[0])), (*C.float)(unsafe.Pointer(&res2[0])))
}

func RunNormal(key_id int, batch_size int, float_data []float32, int_data []int64, res0, res1, res2 []float32) {
	C.RunNormal(C.int(key_id), C.int(batch_size), (*C.float)(unsafe.Pointer(&float_data[0])), (*C.int64_t)(unsafe.Pointer(&int_data[0])), (*C.float)(unsafe.Pointer(&res0[0])), (*C.float)(unsafe.Pointer(&res1[0])), (*C.float)(unsafe.Pointer(&res2[0])))
}

func Run160Model(key_id int, batch_size int, in_data []int64, res0, res1, res2 []float32) {
	C.Run160Model(C.int(key_id), C.int(batch_size), (*C.int64_t)(unsafe.Pointer(&in_data[0])), (*C.float)(unsafe.Pointer(&res0[0])), (*C.float)(unsafe.Pointer(&res1[0])), (*C.float)(unsafe.Pointer(&res2[0])))
}

// reload model.
func Reload(model_dir string, model_file string, params_file string, thread_num int, math_num int, use_lite bool, model_id int) int32 {
	c_model_file := C.CString(model_file)
	c_params_file := C.CString(params_file)
	c_model_dir := C.CString(model_dir)

	reload_id := C.Reload(c_model_dir, c_model_file, c_params_file, C.int(thread_num), C.int(math_num), C.bool(use_lite), C.int(model_id))
	defer func() {
		C.free(unsafe.Pointer(c_model_dir))
		C.free(unsafe.Pointer(c_model_file))
		C.free(unsafe.Pointer(c_params_file))
	}()

	return int32(reload_id)
}
