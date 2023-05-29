TEXT ·_mm256_dot(SB), $0-32
	MOVQ a+0(FP), DI
	MOVQ b+8(FP), SI
	MOVQ n+16(FP), DX
	MOVQ ret+24(FP), CX
	BYTE $0x55                             // pushq	%rbp
	WORD $0x8948; BYTE $0xe5               // movq	%rsp, %rbp
	WORD $0x5641                           // pushq	%r14
	BYTE $0x53                             // pushq	%rbx
	LONG $0xf8e48348                       // andq	$-8, %rsp
	LONG $0x07428d48                       // leaq	7(%rdx), %rax
	WORD $0x8548; BYTE $0xd2               // testq	%rdx, %rdx
	LONG $0xc2490f48                       // cmovnsq	%rdx, %rax
	WORD $0x8949; BYTE $0xc1               // movq	%rax, %r9
	LONG $0x03f9c149                       // sarq	$3, %r9
	LONG $0xf8e08348                       // andq	$-8, %rax
	WORD $0x2948; BYTE $0xc2               // subq	%rax, %rdx
	WORD $0x8545; BYTE $0xc9               // testl	%r9d, %r9d
	JLE  LBB4_1
	LONG $0x0710fcc5                       // vmovups	(%rdi), %ymm0
	LONG $0x0659fcc5                       // vmulps	(%rsi), %ymm0, %ymm0
	LONG $0x20c78348                       // addq	$32, %rdi
	LONG $0x20c68348                       // addq	$32, %rsi
	LONG $0x01f98341                       // cmpl	$1, %r9d
	JE   LBB4_9
	QUAD $0x0007fffffff0b849; WORD $0x0000 // movabsq	$34359738352, %r8
	LONG $0xc8048d4b                       // leaq	(%r8,%r9,8), %rax
	LONG $0x08c88349                       // orq	$8, %r8
	WORD $0x2149; BYTE $0xc0               // andq	%rax, %r8
	LONG $0xff598d45                       // leal	-1(%r9), %r11d
	LONG $0xfe418d41                       // leal	-2(%r9), %eax
	WORD $0xf883; BYTE $0x03               // cmpl	$3, %eax
	JAE  LBB4_16
	WORD $0x8949; BYTE $0xfa               // movq	%rdi, %r10
	WORD $0x8948; BYTE $0xf0               // movq	%rsi, %rax
	JMP  LBB4_5

LBB4_1:
	JMP LBB4_9

LBB4_16:
	WORD $0x8944; BYTE $0xdb // movl	%r11d, %ebx
	WORD $0xe383; BYTE $0xfc // andl	$-4, %ebx
	WORD $0xdbf7             // negl	%ebx
	WORD $0x8949; BYTE $0xfa // movq	%rdi, %r10
	WORD $0x8948; BYTE $0xf0 // movq	%rsi, %rax

LBB4_17:
	LONG $0x107cc1c4; BYTE $0x0a   // vmovups	(%r10), %ymm1
	LONG $0x107cc1c4; WORD $0x2052 // vmovups	32(%r10), %ymm2
	LONG $0x107cc1c4; WORD $0x405a // vmovups	64(%r10), %ymm3
	LONG $0x107cc1c4; WORD $0x6062 // vmovups	96(%r10), %ymm4
	LONG $0x0859f4c5               // vmulps	(%rax), %ymm1, %ymm1
	LONG $0xc158fcc5               // vaddps	%ymm1, %ymm0, %ymm0
	LONG $0x4859ecc5; BYTE $0x20   // vmulps	32(%rax), %ymm2, %ymm1
	LONG $0x5059e4c5; BYTE $0x40   // vmulps	64(%rax), %ymm3, %ymm2
	LONG $0xc158fcc5               // vaddps	%ymm1, %ymm0, %ymm0
	LONG $0xc258fcc5               // vaddps	%ymm2, %ymm0, %ymm0
	LONG $0x4859dcc5; BYTE $0x60   // vmulps	96(%rax), %ymm4, %ymm1
	LONG $0xc158fcc5               // vaddps	%ymm1, %ymm0, %ymm0
	LONG $0x80ea8349               // subq	$-128, %r10
	LONG $0x80e88348               // subq	$-128, %rax
	WORD $0xc383; BYTE $0x04       // addl	$4, %ebx
	JNE  LBB4_17

LBB4_5:
	LONG $0x08708d4d // leaq	8(%r8), %r14
	LONG $0x03c3f641 // testb	$3, %r11b
	JE   LBB4_8
	LONG $0xffc18041 // addb	$-1, %r9b
	LONG $0xc9b60f45 // movzbl	%r9b, %r9d
	LONG $0x03e18341 // andl	$3, %r9d
	LONG $0x05e1c149 // shlq	$5, %r9
	WORD $0xdb31     // xorl	%ebx, %ebx

LBB4_7:
	LONG $0x107cc1c4; WORD $0x1a0c // vmovups	(%r10,%rbx), %ymm1
	LONG $0x0c59f4c5; BYTE $0x18   // vmulps	(%rax,%rbx), %ymm1, %ymm1
	LONG $0xc158fcc5               // vaddps	%ymm1, %ymm0, %ymm0
	LONG $0x20c38348               // addq	$32, %rbx
	WORD $0x3941; BYTE $0xd9       // cmpl	%ebx, %r9d
	JNE  LBB4_7

LBB4_8:
	LONG $0x873c8d4a // leaq	(%rdi,%r8,4), %rdi
	LONG $0x20c78348 // addq	$32, %rdi
	LONG $0xb6348d4a // leaq	(%rsi,%r14,4), %rsi

LBB4_9:
	LONG $0x197de3c4; WORD $0x01c1 // vextractf128	$1, %ymm0, %xmm1
	LONG $0xc058f0c5               // vaddps	%xmm0, %xmm1, %xmm0
	LONG $0x0579e3c4; WORD $0x01c8 // vpermilpd	$1, %xmm0, %xmm1
	LONG $0xc158f8c5               // vaddps	%xmm1, %xmm0, %xmm0
	LONG $0xc816fac5               // vmovshdup	%xmm0, %xmm1
	LONG $0xc158fac5               // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5               // vmovss	%xmm0, (%rcx)
	WORD $0xd285                   // testl	%edx, %edx
	JLE  LBB4_15
	WORD $0x8941; BYTE $0xd0       // movl	%edx, %r8d
	LONG $0xff408d49               // leaq	-1(%r8), %rax
	WORD $0xe283; BYTE $0x03       // andl	$3, %edx
	LONG $0x03f88348               // cmpq	$3, %rax
	JAE  LBB4_18
	WORD $0xc031                   // xorl	%eax, %eax
	JMP  LBB4_12

LBB4_18:
	WORD $0x2949; BYTE $0xd0 // subq	%rdx, %r8
	WORD $0xc031             // xorl	%eax, %eax

LBB4_19:
	LONG $0x0c10fac5; BYTE $0x87   // vmovss	(%rdi,%rax,4), %xmm1
	LONG $0x0c59f2c5; BYTE $0x86   // vmulss	(%rsi,%rax,4), %xmm1, %xmm1
	LONG $0xc158fac5               // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5               // vmovss	%xmm0, (%rcx)
	LONG $0x4c10fac5; WORD $0x0487 // vmovss	4(%rdi,%rax,4), %xmm1
	LONG $0x4c59f2c5; WORD $0x0486 // vmulss	4(%rsi,%rax,4), %xmm1, %xmm1
	LONG $0xc158fac5               // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5               // vmovss	%xmm0, (%rcx)
	LONG $0x4c10fac5; WORD $0x0887 // vmovss	8(%rdi,%rax,4), %xmm1
	LONG $0x4c59f2c5; WORD $0x0886 // vmulss	8(%rsi,%rax,4), %xmm1, %xmm1
	LONG $0xc158fac5               // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5               // vmovss	%xmm0, (%rcx)
	LONG $0x4c10fac5; WORD $0x0c87 // vmovss	12(%rdi,%rax,4), %xmm1
	LONG $0x4c59f2c5; WORD $0x0c86 // vmulss	12(%rsi,%rax,4), %xmm1, %xmm1
	LONG $0xc158fac5               // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5               // vmovss	%xmm0, (%rcx)
	LONG $0x04c08348               // addq	$4, %rax
	WORD $0x3949; BYTE $0xc0       // cmpq	%rax, %r8
	JNE  LBB4_19

LBB4_12:
	WORD $0x8548; BYTE $0xd2 // testq	%rdx, %rdx
	JE   LBB4_15
	LONG $0x86348d48         // leaq	(%rsi,%rax,4), %rsi
	LONG $0x87048d48         // leaq	(%rdi,%rax,4), %rax
	WORD $0xff31             // xorl	%edi, %edi

LBB4_14:
	LONG $0x0c10fac5; BYTE $0xb8 // vmovss	(%rax,%rdi,4), %xmm1
	LONG $0x0c59f2c5; BYTE $0xbe // vmulss	(%rsi,%rdi,4), %xmm1, %xmm1
	LONG $0xc158fac5             // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5             // vmovss	%xmm0, (%rcx)
	LONG $0x01c78348             // addq	$1, %rdi
	WORD $0x3948; BYTE $0xfa     // cmpq	%rdi, %rdx
	JNE  LBB4_14

LBB4_15:
	LONG $0xf0658d48         // leaq	-16(%rbp), %rsp
	BYTE $0x5b               // popq	%rbx
	WORD $0x5e41             // popq	%r14
	BYTE $0x5d               // popq	%rbp
	WORD $0xf8c5; BYTE $0x77 // vzeroupper
	BYTE $0xc3               // retq
