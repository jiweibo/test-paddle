package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sync"
	"test/engine"
	"time"
)

var wg sync.WaitGroup
var mx sync.Mutex

var thread_num = 16
var math_num = 2
var use_lite = true

// var request_num = 40000
var request_num = 16
var max_batch_size = 500

var all_times []time.Duration

var ch = make(chan int, thread_num)
var reload_id int32 = 0

var res0 [][]float32
var res1 [][]float32
var res2 [][]float32

func alloc_space(max_batch_size int, thread_num int) {
	for i := 0; i < thread_num; i++ {
		data0 := make([]float32, max_batch_size)
		res0 = append(res0, data0)

		data1 := make([]float32, max_batch_size)
		res1 = append(res1, data1)

		data2 := make([]float32, max_batch_size)
		res2 = append(res2, data2)
	}
}

func main() {
	engine.InitEngine("models/160inut/", "", "", thread_num, math_num, use_lite)

	// allocate memory.
	alloc_space(max_batch_size, thread_num)
	features := read_json_data()
	batch_size := 500
	fmt.Println("prepare input data done.")

	for i := 0; i < thread_num; i++ {
		ch <- i
	}

	for i := 0; i < request_num; i++ {
		key_id := <-ch
		wg.Add(1)
		go func(key_id int, reload_id int32) {
			start_time := time.Now()

			for n, v := range features {
				engine.SetInputData(reload_id, key_id, n, v, []int32{int32(batch_size), 1})
			}

			engine.Run(reload_id, key_id, res0[key_id], res1[key_id], res2[key_id])
			last := time.Now().Sub(start_time)

			// print avg output.
			print_out_avg(key_id, batch_size)

			// time info.
			mx.Lock()
			all_times = append(all_times, last)

			defer func() {
				wg.Done()
				mx.Unlock()
				ch <- key_id
			}()
		}(key_id, reload_id)
	}

	wg.Wait()
	// engine.TimeInfo(all_times)
}

func read_json_data() map[string][]int64 {
	features := make(map[string][]int64)
	file_bytes, _ := ioutil.ReadFile("data/160_model_input.txt")
	json.Unmarshal(file_bytes, &features)
	return features
}

func print_out_avg(key_id, batch_size int) {
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
