TEXT ·_mm512_dot(SB), $0-32
	MOVQ a+0(FP), DI
	MOVQ b+8(FP), SI
	MOVQ n+16(FP), DX
	MOVQ ret+24(FP), CX
	BYTE $0x55                             // pushq	%rbp
	WORD $0x8948; BYTE $0xe5               // movq	%rsp, %rbp
	WORD $0x5641                           // pushq	%r14
	BYTE $0x53                             // pushq	%rbx
	LONG $0xf8e48348                       // andq	$-8, %rsp
	LONG $0x0f4a8d4c                       // leaq	15(%rdx), %r9
	WORD $0x8548; BYTE $0xd2               // testq	%rdx, %rdx
	LONG $0xca490f4c                       // cmovnsq	%rdx, %r9
	LONG $0x04f9c149                       // sarq	$4, %r9
	WORD $0x8944; BYTE $0xc8               // movl	%r9d, %eax
	WORD $0xe0c1; BYTE $0x04               // shll	$4, %eax
	WORD $0xc229                           // subl	%eax, %edx
	WORD $0x8545; BYTE $0xc9               // testl	%r9d, %r9d
	JLE  LBB4_1
	LONG $0x487cf162; WORD $0x0710         // vmovups	(%rdi), %zmm0
	LONG $0x487cf162; WORD $0x0659         // vmulps	(%rsi), %zmm0, %zmm0
	LONG $0x40c78348                       // addq	$64, %rdi
	LONG $0x40c68348                       // addq	$64, %rsi
	LONG $0x01f98341                       // cmpl	$1, %r9d
	JE   LBB4_9
	WORD $0x894c; BYTE $0xc8               // movq	%r9, %rax
	LONG $0x04e0c148                       // shlq	$4, %rax
	QUAD $0x000fffffffe0b849; WORD $0x0000 // movabsq	$68719476704, %r8
	WORD $0x014c; BYTE $0xc0               // addq	%r8, %rax
	LONG $0x10c88349                       // orq	$16, %r8
	WORD $0x2149; BYTE $0xc0               // andq	%rax, %r8
	LONG $0xff598d45                       // leal	-1(%r9), %r11d
	LONG $0xfe418d41                       // leal	-2(%r9), %eax
	WORD $0xf883; BYTE $0x03               // cmpl	$3, %eax
	JAE  LBB4_18
	WORD $0x8949; BYTE $0xfa               // movq	%rdi, %r10
	WORD $0x8948; BYTE $0xf0               // movq	%rsi, %rax
	JMP  LBB4_5

LBB4_1:
	JMP LBB4_9

LBB4_18:
	WORD $0x8944; BYTE $0xdb // movl	%r11d, %ebx
	WORD $0xe383; BYTE $0xfc // andl	$-4, %ebx
	WORD $0xdbf7             // negl	%ebx
	WORD $0x8949; BYTE $0xfa // movq	%rdi, %r10
	WORD $0x8948; BYTE $0xf0 // movq	%rsi, %rax

LBB4_19:
	LONG $0x487cd162; WORD $0x0a10             // vmovups	(%r10), %zmm1
	LONG $0x487cd162; WORD $0x5210; BYTE $0x01 // vmovups	64(%r10), %zmm2
	LONG $0x487cd162; WORD $0x5a10; BYTE $0x02 // vmovups	128(%r10), %zmm3
	LONG $0x487cd162; WORD $0x6210; BYTE $0x03 // vmovups	192(%r10), %zmm4
	LONG $0x487df262; WORD $0x0898             // vfmadd132ps	(%rax), %zmm0, %zmm1
	LONG $0x486df262; WORD $0x48b8; BYTE $0x01 // vfmadd231ps	64(%rax), %zmm2, %zmm1
	LONG $0x4865f262; WORD $0x48b8; BYTE $0x02 // vfmadd231ps	128(%rax), %zmm3, %zmm1
	LONG $0x487cf162; WORD $0xc128             // vmovaps	%zmm1, %zmm0
	LONG $0x485df262; WORD $0x40b8; BYTE $0x03 // vfmadd231ps	192(%rax), %zmm4, %zmm0
	LONG $0x00c28149; WORD $0x0001; BYTE $0x00 // addq	$256, %r10
	LONG $0x01000548; WORD $0x0000             // addq	$256, %rax
	WORD $0xc383; BYTE $0x04                   // addl	$4, %ebx
	JNE  LBB4_19

LBB4_5:
	LONG $0x10708d4d // leaq	16(%r8), %r14
	LONG $0x03c3f641 // testb	$3, %r11b
	JE   LBB4_8
	LONG $0xffc18041 // addb	$-1, %r9b
	LONG $0xc9b60f45 // movzbl	%r9b, %r9d
	LONG $0x03e18341 // andl	$3, %r9d
	LONG $0x06e1c149 // shlq	$6, %r9
	WORD $0xdb31     // xorl	%ebx, %ebx

LBB4_7:
	LONG $0x487cd162; WORD $0x0c10; BYTE $0x1a // vmovups	(%r10,%rbx), %zmm1
	LONG $0x4875f262; WORD $0x04b8; BYTE $0x18 // vfmadd231ps	(%rax,%rbx), %zmm1, %zmm0
	LONG $0x40c38348                           // addq	$64, %rbx
	WORD $0x3941; BYTE $0xd9                   // cmpl	%ebx, %r9d
	JNE  LBB4_7

LBB4_8:
	LONG $0x873c8d4a // leaq	(%rdi,%r8,4), %rdi
	LONG $0x40c78348 // addq	$64, %rdi
	LONG $0xb6348d4a // leaq	(%rsi,%r14,4), %rsi

LBB4_9:
	LONG $0x48fdf362; WORD $0xc11b; BYTE $0x01 // vextractf64x4	$1, %zmm0, %ymm1
	LONG $0xc058f4c5                           // vaddps	%ymm0, %ymm1, %ymm0
	LONG $0x197de3c4; WORD $0x01c1             // vextractf128	$1, %ymm0, %xmm1
	LONG $0xc058f0c5                           // vaddps	%xmm0, %xmm1, %xmm0
	LONG $0x0579e3c4; WORD $0x01c8             // vpermilpd	$1, %xmm0, %xmm1
	LONG $0xc158f8c5                           // vaddps	%xmm1, %xmm0, %xmm0
	LONG $0xc816fac5                           // vmovshdup	%xmm0, %xmm1
	LONG $0xc158f8c5                           // vaddps	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5                           // vmovss	%xmm0, (%rcx)
	WORD $0xfa83; BYTE $0x07                   // cmpl	$7, %edx
	JLE  LBB4_11
	LONG $0x0f10fcc5                           // vmovups	(%rdi), %ymm1
	LONG $0x0e59f4c5                           // vmulps	(%rsi), %ymm1, %ymm1
	LONG $0x20c78348                           // addq	$32, %rdi
	LONG $0x20c68348                           // addq	$32, %rsi
	LONG $0x197de3c4; WORD $0x01ca             // vextractf128	$1, %ymm1, %xmm2
	LONG $0xc958e8c5                           // vaddps	%xmm1, %xmm2, %xmm1
	LONG $0x0579e3c4; WORD $0x01d1             // vpermilpd	$1, %xmm1, %xmm2
	LONG $0xca58f0c5                           // vaddps	%xmm2, %xmm1, %xmm1
	LONG $0xd116fac5                           // vmovshdup	%xmm1, %xmm2
	LONG $0xca58f2c5                           // vaddss	%xmm2, %xmm1, %xmm1
	LONG $0xc158fac5                           // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5                           // vmovss	%xmm0, (%rcx)
	WORD $0xc283; BYTE $0xf8                   // addl	$-8, %edx

LBB4_11:
	WORD $0xd285             // testl	%edx, %edx
	JLE  LBB4_17
	WORD $0xd289             // movl	%edx, %edx
	LONG $0xff428d48         // leaq	-1(%rdx), %rax
	WORD $0x8941; BYTE $0xd0 // movl	%edx, %r8d
	LONG $0x03e08341         // andl	$3, %r8d
	LONG $0x03f88348         // cmpq	$3, %rax
	JAE  LBB4_20
	WORD $0xc031             // xorl	%eax, %eax
	JMP  LBB4_14

LBB4_20:
	WORD $0xe283; BYTE $0xfc // andl	$-4, %edx
	WORD $0xc031             // xorl	%eax, %eax

LBB4_21:
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
	WORD $0x3948; BYTE $0xc2       // cmpq	%rax, %rdx
	JNE  LBB4_21

LBB4_14:
	WORD $0x854d; BYTE $0xc0 // testq	%r8, %r8
	JE   LBB4_17
	LONG $0x86148d48         // leaq	(%rsi,%rax,4), %rdx
	LONG $0x87048d48         // leaq	(%rdi,%rax,4), %rax
	WORD $0xf631             // xorl	%esi, %esi

LBB4_16:
	LONG $0x0c10fac5; BYTE $0xb0 // vmovss	(%rax,%rsi,4), %xmm1
	LONG $0x0c59f2c5; BYTE $0xb2 // vmulss	(%rdx,%rsi,4), %xmm1, %xmm1
	LONG $0xc158fac5             // vaddss	%xmm1, %xmm0, %xmm0
	LONG $0x0111fac5             // vmovss	%xmm0, (%rcx)
	LONG $0x01c68348             // addq	$1, %rsi
	WORD $0x3949; BYTE $0xf0     // cmpq	%rsi, %r8
	JNE  LBB4_16

LBB4_17:
	LONG $0xf0658d48         // leaq	-16(%rbp), %rsp
	BYTE $0x5b               // popq	%rbx
	WORD $0x5e41             // popq	%r14
	BYTE $0x5d               // popq	%rbp
	WORD $0xf8c5; BYTE $0x77 // vzeroupper
	BYTE $0xc3               // retq
