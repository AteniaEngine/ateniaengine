use atenia_engine::*;

#[test]
fn apx_9_1_gpu_ir_basic_kernel() {
    let mut k = GpuIrKernel::new("vec_add");
    k.params.push(GpuIrParam { name: "a".into(), ty: GpuIrType::Ptr });
    k.params.push(GpuIrParam { name: "b".into(), ty: GpuIrType::Ptr });

    k.body.push(GpuIrStmt::Comment("Load values".into()));
    k.body.push(GpuIrStmt::LocalVar { name: "i".into(), ty: GpuIrType::I32 });

    assert_eq!(k.name, "vec_add");
    assert_eq!(k.params.len(), 2);
}

#[test]
fn apx_9_1_gpu_ir_thread_control() {
    let stmt = GpuIrStmt::ThreadIdxX("tid".into());

    match stmt {
        GpuIrStmt::ThreadIdxX(_) => {}
        _ => panic!("incorrect variant"),
    }
}

#[test]
fn apx_9_1_gpu_ir_loop_structure() {
    let loop_ir = GpuIrStmt::For {
        var: "i".into(),
        start: 0,
        end: 10,
        body: vec![GpuIrStmt::Comment("loop body".into())],
    };

    match loop_ir {
        GpuIrStmt::For { start, end, .. } => {
            assert_eq!(start, 0);
            assert_eq!(end, 10);
        }
        _ => panic!("wrong IR tag"),
    }
}
