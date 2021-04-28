package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strconv"
	"sync"
	"test/engine"
	"time"
)

var wg sync.WaitGroup
var mx sync.Mutex

var thread_num = 16
var math_num = 2
var use_lite = false

// var request_num = 16
var request_num = 40000
var max_batch_size = 500

var all_times []time.Duration

var ch = make(chan int, thread_num)

var float_data [][]float32
var int_data [][]int64
var res0 [][]float32
var res1 [][]float32
var res2 [][]float32

func alloc_space(max_batch_size int, thread_num int) {
	for i := 0; i < thread_num; i++ {
		data1 := make([]float32, max_batch_size*172)
		float_data = append(float_data, data1)

		data2 := make([]int64, max_batch_size*(344-172+10*(358-344)))
		int_data = append(int_data, data2)

		data3 := make([]float32, max_batch_size)
		res0 = append(res0, data3)

		data4 := make([]float32, max_batch_size)
		res1 = append(res1, data4)

		data5 := make([]float32, max_batch_size)
		res2 = append(res2, data5)
	}
}

func main() {
	engine.InitEngine("models/model", "", "", thread_num, math_num, use_lite)

	// allocate memory.
	alloc_space(max_batch_size, thread_num)
	batch_size := 500
	features, features_int := read_json_data()

	fmt.Println("prepare input data done.")

	for i := 0; i < thread_num; i++ {
		ch <- i
	}

	for i := 0; i < request_num; i++ {
		key_id := <-ch
		wg.Add(1)
		go func(key_id int) {
			start_time := time.Now()
			prepare_input(batch_size, key_id, features, features_int)
			engine.RunNormal(key_id, batch_size, float_data[key_id], int_data[key_id], res0[key_id], res1[key_id], res2[key_id])
			last := time.Now().Sub(start_time)

			// print avg output.
			// print_out_avg(key_id, batch_size)

			// time info.
			mx.Lock()
			all_times = append(all_times, last)

			defer func() {
				wg.Done()
				mx.Unlock()
				ch <- key_id
			}()
		}(key_id)
	}

	wg.Wait()
	engine.TimeInfo(all_times)
}

func read_json_data() (map[string][]float32, map[string][]int64) {
	features := make(map[string][]float32)
	features_int := make(map[string][]int64)
	file_bytes, _ := ioutil.ReadFile("data/500input.txt")
	json.Unmarshal(file_bytes, &features)

	for i := 172; i < 358; i++ {
		name := "field_" + strconv.Itoa(i)
		tmp_data := make([]int64, len(features[name]))
		for i, v := range features[name] {
			tmp_data[i] = int64(v)
		}
		features_int[name] = tmp_data
	}
	return features, features_int
}

func prepare_input(batch_size int, thread_id int, features map[string][]float32, features_int map[string][]int64) {
	// field_0 - field_171 is float32 input
	for i := 0; i < 172; i++ {
		name := "field_" + strconv.Itoa(i)
		copy(float_data[thread_id][i*batch_size:(i+1)*batch_size], features[name])
	}

	// field_172 - field_343 is int64 input, [batch_size, 1]
	for i := 172; i < 344; i++ {
		name := "field_" + strconv.Itoa(i)
		copy(int_data[thread_id][(i-172)*batch_size:(i+1-172)*batch_size], features_int[name])
	}

	// field_344 - field_357 is int64[batch_size, 10] input
	base := batch_size * (344 - 172)
	for i := 344; i < 358; i++ {
		name := "field_" + strconv.Itoa(i)
		copy(int_data[thread_id][base+(i-344)*batch_size*10:base+(i+1-344)*batch_size*10], features_int[name])
	}
}

func print_out_avg(key_id int, batch_size int) {
	var avg0 float32 = 0.
	var avg1 float32 = 0.
	var avg2 float32 = 0.
	for i := 0; i < batch_size; i++ {
		avg0 += res0[key_id][i]
		avg1 += res1[key_id][i]
		avg2 += res2[key_id][i]
	}
	fmt.Println(avg0/float32(batch_size), avg1/float32(batch_size), avg2/float32(batch_size))
}
