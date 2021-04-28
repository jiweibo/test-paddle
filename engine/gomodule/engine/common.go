package engine

import (
	"fmt"
	"sort"
	"time"
)

func TimeInfo(all_times []time.Duration) {
	sort.Slice(all_times, func(i, j int) bool {
		return all_times[i] < all_times[j]
	})
	req_percent := []float32{0.5, 0.9, 0.99}
	for _, p := range req_percent {
		idx := int32(float32(len(all_times))*p) - 1
		fmt.Printf("percent %v, cost time %v\n", p, all_times[idx])
	}
	var avg time.Duration = 0
	for _, t := range all_times {
		avg += t
	}
	fmt.Printf("avg time %v\n", float32(avg.Nanoseconds()/1e6)/float32(len(all_times)))
}
