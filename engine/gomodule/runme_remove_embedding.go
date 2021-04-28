package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"sync"
	"test/engine"
	"time"
)

var wg sync.WaitGroup
var mx sync.Mutex

var thread_num = 16
var math_num = 2
var use_lite = false
var request_num = 20000
var max_batch_size = 500
var magic_num = 2404

var all_times []time.Duration

var ch = make(chan int, thread_num)

var in_data [][]float32
var res0 [][]float32
var res1 [][]float32
var res2 [][]float32

func alloc_space(max_batch_size int, thread_num int) {
	for i := 0; i < thread_num; i++ {
		data1 := make([]float32, max_batch_size*magic_num)
		in_data = append(in_data, data1)

		data3 := make([]float32, max_batch_size)
		res0 = append(res0, data3)

		data4 := make([]float32, max_batch_size)
		res1 = append(res1, data4)

		data5 := make([]float32, max_batch_size)
		res2 = append(res2, data5)
	}
}

func main() {
	engine.InitEngine("models/model_prune", "", "", thread_num, math_num, use_lite)

	// allocate memory.
	alloc_space(max_batch_size, thread_num)

	// load embedding and prepare input.
	batch_size := 500
	emb := load_embedding()
	features, features_int := read_json_data()

	fmt.Println("prepare input data done.")

	for i := 0; i < thread_num; i++ {
		ch <- i
	}

	for i := 0; i < request_num; i++ {
		key_id := <-ch
		wg.Add(1)
		go func(key_id int) {
			// fmt.Println(key_id)
			start_time := time.Now()
			prepare_input(emb, batch_size, key_id, features, features_int)
			engine.Run(key_id, batch_size, in_data[key_id], batch_size*magic_num, res0[key_id], res1[key_id], res2[key_id])
			last := time.Now().Sub(start_time)

			// print avg output.
			// var avg0 float32 = 0.
			// var avg1 float32 = 0.
			// var avg2 float32 = 0.
			// for i := 0; i < batch_size; i++ {
			// 	avg0 += res0[key_id][i]
			// 	avg1 += res1[key_id][i]
			// 	avg2 += res2[key_id][i]
			// }
			// fmt.Println(last, avg0/float32(batch_size), avg1/float32(batch_size), avg2/float32(batch_size))

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

func load_embedding() map[int64][]float32 {
	embedding := make(map[int64][]float32)
	file_bytes, _ := ioutil.ReadFile("models/model_prune/SparseFeatFactors")
	for _, line := range strings.Split(string(file_bytes), "\n") {
		if line == "" {
			continue
		}
		columns := strings.Split(line, "\t")
		id_val, _ := strconv.ParseInt(columns[0], 10, 64)
		embx := make([]float32, 12)
		w := strings.Split(string(columns[4]), ",")
		for i := 0; i < 12; i++ {
			val, _ := strconv.ParseFloat(w[i], 32)
			embx[i] = float32(val)
		}
		embedding[id_val] = embx
	}
	return embedding
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

func prepare_input(emb map[int64][]float32, batch_size int, thread_id int, features map[string][]float32, features_int map[string][]int64) {
	for b := 0; b < batch_size; b++ {
		base := b * magic_num

		for i := 0; i < 172; i++ {
			name := "field_" + strconv.Itoa(i)
			copy(in_data[thread_id][base+i:base+i+1], features[name][b:b+1])
		}

		base += 172
		for i := 172; i < 344; i++ {
			name := "field_" + strconv.Itoa(i)
			copy(in_data[thread_id][base+(i-172)*12:base+(i+1-172)*12], emb[features_int[name][b]])
		}

		base += 172 * 12
		for i := 344; i < 358; i++ {
			// reduce sum
			name := "field_" + strconv.Itoa(i)
			sum := make([]float32, 12)
			for _, v := range features_int[name][b*10 : (b+1)*10] {
				for j := 0; j < 12; j++ {
					sum[j] += emb[v][j]
				}
			}
			copy(in_data[thread_id][base+(i-344)*12:base+(i+1-344)*12], sum)
		}
	}
}
