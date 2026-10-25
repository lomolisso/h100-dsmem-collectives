# Verbatim excerpts (FlashAttention-4, CuTe-DSL, sm100) backing kernels/flash_bwd_2cta_ds_exchange__fa4.cu.
# Source commit: third_party/flash-attention @ 98c1cee. Not runnable on its own; for traceability only.
# ---- copy_utils.py (mapa, st.async, cp.async.bulk s2cluster wrappers) lines 147-239 ----
@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote_fp32x4(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        "mov.f32 abcd.x, $2;\n\t"
        "mov.f32 abcd.y, $3;\n\t"
        "mov.f32 abcd.z, $4;\n\t"
        "mov.f32 abcd.w, $5;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], abcd, [$1];\n\t"
        "}\n",
        "r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_bulk_s2cluster(
    smem_src_ptr: cute.Pointer,
    smem_dst_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
):
    smem_src_ptr_i32 = smem_src_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_dst_ptr_i32 = set_block_rank(
        smem_dst_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_dst_ptr_i32,
            smem_src_ptr_i32,
            mbar_ptr_i32,
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
# ---- flash_bwd_sm100.py lines 598-620 ----
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        if self.acc_m64:
            self._check_dS_exchange_layouts()
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
# ---- flash_bwd_sm100.py lines 955-960 ----
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        # TileScheduler = SingleTileScheduler
# ---- flash_bwd_sm100.py lines 1060-1070 ----
                tmem_holding_buf: Int32
                tmem_dealloc_mbar: cutlass.Int64

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dQaccum_empty_mbar_ptr: cutlass.Int64

# ---- flash_bwd_sm100.py lines 1435-1446 ----
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

# ---- flash_bwd_sm100.py lines 1955-2008 ----
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = self.kv_head_of(head_idx)

            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q) or m_block_min < m_block_max
            )
            num_iters = m_block_max - m_block_min
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                num_iters = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block // self.kv_subtile_factor,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = num_iters > Int32(0)

            if process_tile:
                for _ in cutlass.range(num_iters, unroll=1):
                    # Wait for dS_xchg from peer CTA
                    cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)

                    # Arrive on MMA leader warp
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))

                    dS_cluster_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
# ---- flash_bwd_sm100.py lines 2905-2925 ----
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        # Note: if dS is written to tmem, P must be written to tmem
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1

                    # signal to the epilogue that dV is ready
# ---- flash_bwd_sm100.py lines 3565-3580 ----
            tRS_sdS_dK = self.split_wg(thr_copy_r2s.partition_D(sdS_dK_epi), wg_idx, num_wg)
            # Q-column half this thread's chunk belongs to (from its partition coordinates); the
            # CTA keeps the half it owns for dQ (the peer's half reaches it from sdS_dK, see the
            # exchange copy below).
            q_half = tScS_t2r[0][1] // (self.tile_m // 2)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_empty_phase = Int32(1)
        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)
        if const_expr(self.use_2cta_instrs):
            peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
            sdS_xchg_src = self._dS_exchange_source(sdS_xchg, sdS_dK, peer_cta_rank_in_cluster)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
# ---- flash_bwd_sm100.py lines 3900-3998 ----
                        )
                        # Zero out OOB positions (kv_idx >= seqlen_k) after score_mod_bwd
                        for i in cutlass.range(cute.size(tdPrdP_cur), unroll_full=True):
                            kv_idx = tScS_idx_cur[i][0]
                            tdPrdP_cur[i] = 0.0 if kv_idx >= seqlen.seqlen_k else tdPrdP_cur[i]

                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)
                        if const_expr(self.use_2cta_instrs and not self.acc_m64):
                            tdPrdS_xchg = cute.make_fragment_like(tdPrdS_cvt, self.ds_dtype)

                    if const_expr(self.acc_m64):
                        # dS^T for the dK MMA: this CTA's rows, both Q-column halves; the half
                        # the peer needs for its dQ MMA is exchanged straight from here.
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS_dK[None, 0, 0, 0])
                        # dS for this CTA's own dQ MMA: the Q-column half it owns.
                        if q_half == cta_rank_in_cluster:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, 0, 0, 0])
                    else:
                        # RMEM->TMEM: always write to TMEM for MMA
                        if const_expr(not self.use_smem_dS_for_mma_dK or self.use_2cta_instrs):
                            tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                            cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                        # RMEM->SMEM: For 2-CTA, keep exchange stage in registers, write
                        # non-exchange to sdS
                        if const_expr(self.use_2cta_instrs):
                            if exchange_stage == stage:
                                cute.autovec_copy(tdPrdS_cvt, tdPrdS_xchg)
                            else:
                                cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                        else:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                if const_expr(self.acc_m64):
                    # dS^T is in SMEM: make this warp's stores visible to the async proxy before
                    # the per-warp arrival the MMA warp waits on.
                    cute.arch.fence_view_async_shared()
                elif const_expr(not self.use_smem_dS_for_mma_dK):
                    cute.arch.fence_view_async_tmem_store()

                if const_expr(self.use_2cta_instrs):
                    # use pipeline_dP to signal the dS store (TMEM, or SMEM with acc_m64)
                    with cute.arch.elect_one():
                        pipeline_dP.consumer_release(consumer_state_S_P_dP)
                consumer_state_S_P_dP.advance()

                # After the loop: copy exchange registers to sdS_xchg buffer
                if const_expr(sdS_xchg is not None):
                    # sdS_xchg aliases sdQaccum: wait until the dQ reduce has read it
                    if const_expr(dQaccum_empty_mbar_ptr is not None):
                        cute.arch.mbarrier_wait(
                            dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase
                        )
                    cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

                cute.arch.fence_view_async_shared()
                # The dS and dPsum mbarriers expect one arrival per compute warp
                if const_expr(self.split_P_dS):
                    cute.arch.sync_warp()
                else:
                    self.compute_sync_barrier.arrive_and_wait()
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the barrier or sync_warp before this
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                # pipelined 2CTA schedule: pipeline_dS also signals S tmem load completion so is deferred
                if const_expr(self.cta_schedule != CtaSchedule.TWO_CTA_PIPELINED):
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

                # 2-CTA: DSMEM copy of the exchanged dS half to the peer's sdS buffer
                if const_expr(self.use_2cta_instrs):
                    stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                    stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
                    if tidx == 0:
                        smem_src_ptr = sdS_xchg_src.iterator
                        # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                        smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )
                        copy_utils.cpasync_bulk_s2cluster(
                            smem_src_ptr,
                            smem_dst_ptr,
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )

            # Final signal for dS smem store completion
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                if process_tile:
                    with cute.arch.elect_one():
# ---- flash_bwd_sm100.py lines 4200-4220 ----
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        # For 2-CTA: reduce_stage = dQaccum_reduce_stage_t2r / cta_group_size
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, (
            "dQaccum t2r reduce stage mismatch"
        )
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        # 2-CTA: CTA 0 -> (M/2, D) (stage 0, 1) & CTA 1 -> (M/2, D) (stage 2, 3)
        stage_offset = (
            expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
