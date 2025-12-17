use atenia_engine::apx3_9::op_router::{route, ExecTarget};
use atenia_engine::amg::nodes::NodeType;

#[test]
fn test_router_basic() {
    assert_eq!(route(&NodeType::Add, &[4, 4]), ExecTarget::CPU);
    assert_eq!(
        route(&NodeType::RmsNorm, &[1, 16, 64]),
        ExecTarget::CpuOptimized
    );
    assert_eq!(route(&NodeType::Linear, &[32, 32]), ExecTarget::Auto);
}
