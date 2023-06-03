	.text
	.file	"mul_avx2.c"
	.globl	_mm256_mul_to           # -- Begin function _mm256_mul_to
	.p2align	4, 0x90
	.type	_mm256_mul_to,@function
_mm256_mul_to:                          # @_mm256_mul_to
# %bb.0:
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%rbx
	andq	$-8, %rsp
	leaq	7(%rcx), %rax
	testq	%rcx, %rcx
	cmovnsq	%rcx, %rax
	movq	%rax, %r9
	sarq	$3, %r9
	andq	$-8, %rax
	subq	%rax, %rcx
	testl	%r9d, %r9d
	jle	.LBB0_6
# %bb.1:
	leal	-1(%r9), %eax
	movl	%r9d, %r8d
	andl	$3, %r8d
	cmpl	$3, %eax
	jb	.LBB0_4
# %bb.2:
	movl	%r8d, %eax
	subl	%r9d, %eax
	.p2align	4, 0x90
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
	vmovups	(%rdi), %ymm0
	vmulps	(%rsi), %ymm0, %ymm0
	vmovups	%ymm0, (%rdx)
	vmovups	32(%rdi), %ymm0
	vmulps	32(%rsi), %ymm0, %ymm0
	vmovups	%ymm0, 32(%rdx)
	vmovups	64(%rdi), %ymm0
	vmulps	64(%rsi), %ymm0, %ymm0
	vmovups	%ymm0, 64(%rdx)
	vmovups	96(%rdi), %ymm0
	vmulps	96(%rsi), %ymm0, %ymm0
	vmovups	%ymm0, 96(%rdx)
	subq	$-128, %rdi
	subq	$-128, %rsi
	subq	$-128, %rdx
	addl	$4, %eax
	jne	.LBB0_3
.LBB0_4:
	testl	%r8d, %r8d
	je	.LBB0_6
	.p2align	4, 0x90
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	vmovups	(%rdi), %ymm0
	vmulps	(%rsi), %ymm0, %ymm0
	vmovups	%ymm0, (%rdx)
	addq	$32, %rdi
	addq	$32, %rsi
	addq	$32, %rdx
	addl	$-1, %r8d
	jne	.LBB0_5
.LBB0_6:
	testl	%ecx, %ecx
	jle	.LBB0_18
# %bb.7:
	movl	%ecx, %r8d
	cmpq	$31, %r8
	ja	.LBB0_13
# %bb.8:
	xorl	%eax, %eax
	jmp	.LBB0_9
.LBB0_13:
	leaq	(%rdx,%r8,4), %rax
	leaq	(%rdi,%r8,4), %r9
	leaq	(%rsi,%r8,4), %r10
	cmpq	%r9, %rdx
	setb	%r11b
	cmpq	%rax, %rdi
	setb	%bl
	cmpq	%r10, %rdx
	setb	%r9b
	cmpq	%rax, %rsi
	setb	%r10b
	xorl	%eax, %eax
	testb	%bl, %r11b
	jne	.LBB0_9
# %bb.14:
	andb	%r10b, %r9b
	jne	.LBB0_9
# %bb.15:
	movl	%ecx, %r9d
	andl	$31, %r9d
	movq	%r8, %rax
	subq	%r9, %rax
	xorl	%r10d, %r10d
	.p2align	4, 0x90
.LBB0_16:                               # =>This Inner Loop Header: Depth=1
	vmovups	(%rdi,%r10,4), %ymm0
	vmovups	32(%rdi,%r10,4), %ymm1
	vmovups	64(%rdi,%r10,4), %ymm2
	vmovups	96(%rdi,%r10,4), %ymm3
	vmulps	(%rsi,%r10,4), %ymm0, %ymm0
	vmulps	32(%rsi,%r10,4), %ymm1, %ymm1
	vmulps	64(%rsi,%r10,4), %ymm2, %ymm2
	vmulps	96(%rsi,%r10,4), %ymm3, %ymm3
	vmovups	%ymm0, (%rdx,%r10,4)
	vmovups	%ymm1, 32(%rdx,%r10,4)
	vmovups	%ymm2, 64(%rdx,%r10,4)
	vmovups	%ymm3, 96(%rdx,%r10,4)
	addq	$32, %r10
	cmpq	%r10, %rax
	jne	.LBB0_16
# %bb.17:
	testq	%r9, %r9
	je	.LBB0_18
.LBB0_9:
	subl	%eax, %ecx
	movq	%rax, %r9
	notq	%r9
	addq	%r8, %r9
	andq	$3, %rcx
	je	.LBB0_11
	.p2align	4, 0x90
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	vmovss	(%rdi,%rax,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	vmulss	(%rsi,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rdx,%rax,4)
	addq	$1, %rax
	addq	$-1, %rcx
	jne	.LBB0_10
.LBB0_11:
	cmpq	$3, %r9
	jb	.LBB0_18
	.p2align	4, 0x90
.LBB0_12:                               # =>This Inner Loop Header: Depth=1
	vmovss	(%rdi,%rax,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	vmulss	(%rsi,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rdx,%rax,4)
	vmovss	4(%rdi,%rax,4), %xmm0   # xmm0 = mem[0],zero,zero,zero
	vmulss	4(%rsi,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, 4(%rdx,%rax,4)
	vmovss	8(%rdi,%rax,4), %xmm0   # xmm0 = mem[0],zero,zero,zero
	vmulss	8(%rsi,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, 8(%rdx,%rax,4)
	vmovss	12(%rdi,%rax,4), %xmm0  # xmm0 = mem[0],zero,zero,zero
	vmulss	12(%rsi,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, 12(%rdx,%rax,4)
	addq	$4, %rax
	cmpq	%rax, %r8
	jne	.LBB0_12
.LBB0_18:
	leaq	-8(%rbp), %rsp
	popq	%rbx
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end0:
	.size	_mm256_mul_to, .Lfunc_end0-_mm256_mul_to
                                        # -- End function
	.ident	"clang version 10.0.0-4ubuntu1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
