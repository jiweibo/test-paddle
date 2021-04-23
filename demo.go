package main

import (
	"io/ioutil"
	"encoding/json"
	"fmt"
	"reflect"
	//"test-paddle/data"
	"test-paddle/paddle"
	"time"
        "sync"
	"sort"
	"strconv"
)

var wg sync.WaitGroup
var mx sync.Mutex
var CH = make(chan *paddle.Predictor, 100)

var request_num int = 5000
var thread_num int = 16
var all_times []time.Duration

func GetNewPredictor() *paddle.Predictor {
	config := paddle.NewAnalysisConfig()
	config.SetModel("357model", "")
	//config.SetModel("tf_model", "")
	config.DisableGlogInfo()
	config.SwitchUseFeedFetchOps(false)
        config.SetCpuMathLibraryNumThreads(2)
	predictor := paddle.NewPredictor(config)
	config.MkldnnQuantizerEnabled()
        
	return predictor
}

func main() {

	for i := 0; i < thread_num; i++ {
		CH <- GetNewPredictor()
	}

	features := make(map[string][]float32)
	file_bytes, _ := ioutil.ReadFile("data/500input.txt")
	json.Unmarshal(file_bytes, &features)
	// field_0 - field_171  is float32 input
	// field_172 - field_357  is int64 input
	features_int := make(map[string][]int64)
	for i := 172; i < 358; i++ {
		name := "field_" + strconv.Itoa(i)
		tmp_data := make([]int64, len(features[name]), len(features[name]))
		for i, v := range features[name] {
			tmp_data[i] = int64(v)
		}
		features_int[name] = tmp_data
	}

	for i := 0; i < request_num; i++ {
		predict := <-CH
                wg.Add(1)
		go func(ch chan *paddle.Predictor, p *paddle.Predictor) {
                        t1 := time.Now()
			defer func() {
				ch <- p
                                wg.Done()
			}()
			inputs := p.GetInputTensors()
			//for i, input := range inputs {
			for j := 0; j < 358; j++ {
				input := inputs[j]
				if j < 172 {
					input.SetValue(features[input.Name()])
					input.Reshape([]int32{int32(len(features[input.Name()])), 1})
				} else {
					input.SetValue(features_int[input.Name()])
					if j < 344 {
						input.Reshape([]int32{int32(len(features_int[input.Name()])), 1})
					} else {
						input.Reshape([]int32{int32(len(features_int[input.Name()])/10), 10})
					}
				}
				p.SetZeroCopyInput(input)
			}
			p.ZeroCopyRun()
			outputs := p.GetOutputTensors()

			//output := outputs[2]
			output := outputs[0]
			p.GetZeroCopyOutput(output)

			outputVal := output.Value()
			value := reflect.ValueOf(outputVal)
			tmp := value.Interface().([][]float32)

			var result []float64
			for _, v := range tmp {
				result = append(result, float64(v[0]))
			}
                        sub:=time.Now().Sub(t1)
                        mx.Lock()
                        defer mx.Unlock()
                        all_times = append(all_times, sub)
			//fmt.Printf("time is %v\n", sub)

			//fmt.Printf("result = %+v", result)
		}(CH, predict)
	}
        wg.Wait()

	sort.Slice(all_times, func(i, j int) bool {
		return all_times[i] < all_times[j]
	})
        req_percent := []float32{0.5, 0.9, 0.99}
	for _, p := range req_percent{
		idx := int32(float32(len(all_times)) * p) - 1
                fmt.Printf("percent %v, cost time %v\n", p, all_times[idx])
	}
        var avg time.Duration = 0
        for _, t := range all_times {
		avg += t
	}
	fmt.Printf("avg time %v\n", float32(avg.Nanoseconds() / 1e6)/float32(len(all_times)))
}
