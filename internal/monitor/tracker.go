// Package monitor provides concurrent-safe per-stage latency tracking and
// Go runtime snapshot collection. Zero external dependencies.
//
// Intended for use by the Vox pipeline (and Brain in future) to record named
// stage durations and expose running statistics for the debug TUI.
package monitor

import (
	"runtime"
	"sync"
	"time"
)

// Stats holds running statistics for a named pipeline stage.
type Stats struct {
	Last  time.Duration
	Min   time.Duration
	Max   time.Duration
	Avg   time.Duration
	Count int64
}

// Snapshot is a point-in-time view of all stages plus Go runtime metrics.
type Snapshot struct {
	Stages      map[string]Stats
	Goroutines  int
	GOMAXPROCS  int
	HeapAllocMB float64
	HeapSysMB   float64
	NextGCMB    float64
	NumGC       uint32
	Timestamp   time.Time
}

// stageData tracks running totals for one stage.
type stageData struct {
	last  time.Duration
	min   time.Duration
	max   time.Duration
	total time.Duration
	count int64
}

// Tracker is a concurrent-safe per-stage latency recorder.
// Use New() to create a usable Tracker.
type Tracker struct {
	mu     sync.RWMutex
	stages map[string]*stageData
}

// New returns a new Tracker ready for use.
func New() *Tracker {
	return &Tracker{
		stages: make(map[string]*stageData),
	}
}

// Record records a duration for the named stage, updating running statistics.
// Safe to call from any goroutine.
func (t *Tracker) Record(stage string, d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()

	s, ok := t.stages[stage]
	if !ok {
		s = &stageData{min: d, max: d}
		t.stages[stage] = s
	}
	s.last = d
	s.total += d
	s.count++
	if d < s.min {
		s.min = d
	}
	if d > s.max {
		s.max = d
	}
}

// Snapshot returns a point-in-time copy of all stage stats plus Go runtime metrics.
// Calls runtime.ReadMemStats which briefly stops the world — call at most a few
// times per second (e.g. once per 100ms TUI tick).
func (t *Tracker) Snapshot() Snapshot {
	t.mu.RLock()
	stages := make(map[string]Stats, len(t.stages))
	for name, s := range t.stages {
		var avg time.Duration
		if s.count > 0 {
			avg = s.total / time.Duration(s.count)
		}
		stages[name] = Stats{
			Last:  s.last,
			Min:   s.min,
			Max:   s.max,
			Avg:   avg,
			Count: s.count,
		}
	}
	t.mu.RUnlock()

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	return Snapshot{
		Stages:      stages,
		Goroutines:  runtime.NumGoroutine(),
		GOMAXPROCS:  runtime.GOMAXPROCS(0),
		HeapAllocMB: float64(ms.HeapAlloc) / (1024 * 1024),
		HeapSysMB:   float64(ms.HeapSys) / (1024 * 1024),
		NextGCMB:    float64(ms.NextGC) / (1024 * 1024),
		NumGC:       ms.NumGC,
		Timestamp:   time.Now(),
	}
}
