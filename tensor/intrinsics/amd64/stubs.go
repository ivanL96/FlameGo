package amd64

import "unsafe"

func add(a, b int64) int64

// implemented in /intrinsics/amd64/dot_avx_amd64.s
//go:noescape
func _mm256_dot(a, b, n, ret unsafe.Pointer)
