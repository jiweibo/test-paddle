package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"sort"
	"sync"
	"time"

	pd "github.com/jiweibo/paddle/paddle/fluid/inference/goapi"
)

var wg sync.WaitGroup
var mx sync.Mutex

var threadNum = 16

// var requestNum = 4000000
// var requestNum = 1000
var requestNum = 400000

var times = make([]time.Duration, requestNum)
var idx int = 0

var ch = make(chan int, threadNum)

var res0 [][]float32
var res1 [][]float32
var res2 [][]float32
var maxBatchSize = 500

var inHandles []map[string]*pd.Tensor
var outHandles []map[string]*pd.Tensor

func allocSpace(maxBatchSize int, threadNum int) {
	for i := 0; i < threadNum; i++ {
		data0 := make([]float32, maxBatchSize)
		res0 = append(res0, data0)

		data1 := make([]float32, maxBatchSize)
		res1 = append(res1, data1)

		data2 := make([]float32, maxBatchSize)
		res2 = append(res2, data2)
	}
}

func main() {
	allocSpace(maxBatchSize, threadNum)

	config := pd.NewConfig()
	config.SetModelDir("models/160input")
	config.SetCpuMathLibraryNumThreads(2)

	mainPredictor := pd.NewPredictor(config)
	predictors := []*pd.Predictor{}
	predictors = append(predictors, mainPredictor)
	for i := 0; i < threadNum-1; i++ {
		predictors = append(predictors, mainPredictor.Clone())
	}

	for i := 0; i < threadNum; i++ {
		ch <- i
	}

	outNames := predictors[0].GetOutputNames()

	inHandles = make([]map[string]*pd.Tensor, threadNum)
	outHandles = make([]map[string]*pd.Tensor, threadNum)
	for i := 0; i < threadNum; i++ {
		inHandles[i] = make(map[string]*pd.Tensor)
		inNames := predictors[i].GetInputNames()
		for _, n := range inNames {
			inHandles[i][n] = predictors[i].GetInputHandle(n)
		}
		outHandles[i] = make(map[string]*pd.Tensor)
		for _, n := range outNames {
			outHandles[i][n] = predictors[i].GetOutputHandle(n)
		}
	}

	features := parseJson()

	for i := 0; i < requestNum; i++ {
		keyId := <-ch
		wg.Add(1)
		go func(keyId int) {
			start := time.Now()

			for n, v := range features {
				inHandles[keyId][n].Reshape([]int32{int32(len(v)), 1})
				inHandles[keyId][n].CopyFromCpu(v)
			}
			predictors[keyId].Run()

			outHandles[keyId][outNames[0]].CopyToCpu(res0[keyId])
			outHandles[keyId][outNames[1]].CopyToCpu(res1[keyId])
			outHandles[keyId][outNames[2]].CopyToCpu(res2[keyId])

			last := time.Now().Sub(start)

			// printOutAvg(res0[keyId], res1[keyId], outData2[keyId])

			mx.Lock()
			times[idx] = last
			idx += 1

			defer func() {
				wg.Done()
				mx.Unlock()
				ch <- keyId
			}()
		}(keyId)
	}

	wg.Wait()
	timeInfo(times)
}

func parseJson() map[string][]int64 {
	features := make(map[string][]int64)
	file_bytes, _ := ioutil.ReadFile("data/160_model_input.txt")
	json.Unmarshal(file_bytes, &features)
	return features
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}

func printOutAvg(out0, out1, out2 []float32) {
	var avg0 float32 = 0
	var avg1 float32 = 0
	var avg2 float32 = 0
	for i := 0; i < len(out0); i++ {
		avg0 += out0[i]
		avg1 += out1[i]
		avg2 += out2[i]
	}
	fmt.Println(avg0/float32(len(out0)), avg1/float32(len(out0)), avg2/float32(len(out0)))
}

func timeInfo(times []time.Duration) {
	if len(times) == 1 {
		log.Printf("Only 1 time:%+v\n", times[0])
		return
	}
	sort.Slice(times, func(i, j int) bool {
		return times[i] < times[j]
	})
	req_percent := []float32{0.5, 0.9, 0.95, 0.99}
	for _, p := range req_percent {
		idx := int32(float32(len(times))*p) - 1
		log.Printf("percent %v, cost time %v\n", p, times[idx])
	}
	var avg time.Duration = 0
	for _, t := range times {
		avg += t
	}
	log.Printf("avg time %vms\n", float32(avg.Nanoseconds()/1e6)/float32(len(times)))
}
