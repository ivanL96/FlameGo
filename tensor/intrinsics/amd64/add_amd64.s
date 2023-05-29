// go:build amd64
// add_amd64.s

TEXT ·add(SB), 0, $0-16
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	ADDQ AX, BX
	MOVQ BX, ret+16(FP)
	RET
