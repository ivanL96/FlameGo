package amd64

import "unsafe"

// go install github.com/gorse-io/goat@latest
// implemented in /internal/intrinsics/amd64/

//go:noescape
func _mm256_dot(a, b, n, ret unsafe.Pointer)

//go:noescape
func _mm512_dot(a, b, n, ret unsafe.Pointer)
