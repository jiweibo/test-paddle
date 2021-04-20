package main

import (
	"encoding/json"
	"fmt"
	"reflect"
	"test-paddle/data"
	"test-paddle/paddle"
	"time"
)

func GetNewPredictor() *paddle.Predictor {
	config := paddle.NewAnalysisConfig()
	config.SetModel("model/paddle_esmm_model_v1", "")
	// 输出模型路径
	config.DisableGlogInfo()
	config.SwitchUseFeedFetchOps(false)
	config.SwitchSpecifyInputNames(true)
	config.SwitchIrOptim(false)
	predictor := paddle.NewPredictor(config)
	config.MkldnnQuantizerEnabled()

	return predictor
}

var CH = make(chan *paddle.Predictor, 100)

func main() {



	for i := 0; i < 16; i++ {
		CH <- GetNewPredictor()
	}

	//go func() {
	//	time.Sleep(3 * time.Second)
	//	CH2 := make(chan *paddle.Predictor, 100)
	//	for i := 0; i < 30; i++ {
	//
	//		CH2 <- GetNewPredictor()
	//	}
	//	CH = CH2
	//}()

	features := make(map[string][]int64)
	json.Unmarshal([]byte(data.TestData), &features)


	for i := 0; i < 10000; i++ {
		fmt.Printf("i = %+v \n", i)
		predict := <-CH
		go func(ch chan *paddle.Predictor, p *paddle.Predictor) {

			start := time.Now()
			defer func() {
				ch <- p
			}()
			inputs := p.GetInputTensors()
			for _, input := range inputs {
				input.SetValue(features[input.Name()])
				input.Reshape([]int32{int32(len(features[input.Name()])), 1})
				p.SetZeroCopyInput(input)
			}
			p.ZeroCopyRun()
			outputs := p.GetOutputTensors()

			output := outputs[2]
			p.GetZeroCopyOutput(output)

			outputVal := output.Value()
			value := reflect.ValueOf(outputVal)
			tmp := value.Interface().([][]float32)

			var result []float64
			for _, v := range tmp {
				result = append(result, float64(v[0]))
			}


			fmt.Printf("time = %+v", time.Now().Sub(start))
		}(CH, predict)

	}

}
