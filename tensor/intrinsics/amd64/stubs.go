package amd64

import "unsafe"

func add(a, b int64) int64

func _mm256_dot(a, b, n, ret unsafe.Pointer)
