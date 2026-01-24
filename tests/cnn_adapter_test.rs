#![allow(dead_code)]

use atenia_engine::v17;

use v17::cnn::conv2d::AbortFlag;
use v17::cnn::cnn_adapter::{
    CNNExecutionAdapter,
    CNNGraph,
    CNNLayer,
    CNNLayerKind,
    CNNPlanStepKind,
    CNNAdapterError,
};

#[test]
fn cnn_plan_is_built_with_expected_steps() {
    let graph = CNNGraph {
        layers: vec![
            CNNLayer { name: "conv".to_string(), kind: CNNLayerKind::Conv2D },
            CNNLayer { name: "bias".to_string(), kind: CNNLayerKind::Bias },
            CNNLayer { name: "relu".to_string(), kind: CNNLayerKind::ReLU },
            CNNLayer { name: "pool".to_string(), kind: CNNLayerKind::MaxPool2D },
        ],
    };

    let flag = AbortFlag::new();
    let plan = CNNExecutionAdapter::build_plan(&graph, &flag).unwrap();

    assert_eq!(plan.steps.len(), 4);
    assert_eq!(plan.steps[0].kind, CNNPlanStepKind::Conv2D);
    assert_eq!(plan.steps[1].kind, CNNPlanStepKind::Bias);
    assert_eq!(plan.steps[2].kind, CNNPlanStepKind::ReLU);
    assert_eq!(plan.steps[3].kind, CNNPlanStepKind::MaxPool2D);
}

#[test]
fn cnn_steps_are_ordered_and_abortable() {
    let graph = CNNGraph {
        layers: vec![
            CNNLayer { name: "conv".to_string(), kind: CNNLayerKind::Conv2D },
            CNNLayer { name: "relu".to_string(), kind: CNNLayerKind::ReLU },
        ],
    };

    let flag = AbortFlag::new();
    let plan = CNNExecutionAdapter::build_plan(&graph, &flag).unwrap();

    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps[0].kind, CNNPlanStepKind::Conv2D);
    assert_eq!(plan.steps[1].kind, CNNPlanStepKind::ReLU);

    // Abort before building a plan should yield an error.
    let mut aborted = AbortFlag::new();
    aborted.abort();
    let result = CNNExecutionAdapter::build_plan(&graph, &aborted);
    assert!(matches!(result, Err(CNNAdapterError::Aborted)));
}

#[test]
fn cnn_adapter_respects_execution_contract() {
    let graph = CNNGraph {
        layers: vec![
            CNNLayer { name: "conv".to_string(), kind: CNNLayerKind::Conv2D },
            CNNLayer { name: "bias".to_string(), kind: CNNLayerKind::Bias },
            CNNLayer { name: "relu".to_string(), kind: CNNLayerKind::ReLU },
        ],
    };

    let flag = AbortFlag::new();
    let plan = CNNExecutionAdapter::build_plan(&graph, &flag).unwrap();

    // Minimal contract: plan is non-empty and only uses CNN-related step kinds.
    assert!(!plan.steps.is_empty());
    for step in &plan.steps {
        match step.kind {
            CNNPlanStepKind::Conv2D | CNNPlanStepKind::Bias | CNNPlanStepKind::ReLU | CNNPlanStepKind::MaxPool2D => {},
        }
    }
}

#[test]
fn invalid_cnn_graph_yields_explicit_error() {
    // Empty graph.
    let empty_graph = CNNGraph { layers: vec![] };
    let flag = AbortFlag::new();
    let r = CNNExecutionAdapter::build_plan(&empty_graph, &flag);
    assert!(matches!(r, Err(CNNAdapterError::InvalidGraph(_))));

    // Bias before Conv2D.
    let bad_graph = CNNGraph {
        layers: vec![
            CNNLayer { name: "bias".to_string(), kind: CNNLayerKind::Bias },
            CNNLayer { name: "conv".to_string(), kind: CNNLayerKind::Conv2D },
        ],
    };
    let r2 = CNNExecutionAdapter::build_plan(&bad_graph, &flag);
    assert!(matches!(r2, Err(CNNAdapterError::InvalidGraph(_))));

    // Multiple Conv2D layers not supported in minimal adapter.
    let multi_conv = CNNGraph {
        layers: vec![
            CNNLayer { name: "conv1".to_string(), kind: CNNLayerKind::Conv2D },
            CNNLayer { name: "conv2".to_string(), kind: CNNLayerKind::Conv2D },
        ],
    };
    let r3 = CNNExecutionAdapter::build_plan(&multi_conv, &flag);
    assert!(matches!(r3, Err(CNNAdapterError::InvalidGraph(_))));
}

#[test]
fn cnn_adapter_is_deterministic() {
    let graph = CNNGraph {
        layers: vec![
            CNNLayer { name: "conv".to_string(), kind: CNNLayerKind::Conv2D },
            CNNLayer { name: "bias".to_string(), kind: CNNLayerKind::Bias },
            CNNLayer { name: "relu".to_string(), kind: CNNLayerKind::ReLU },
            CNNLayer { name: "pool".to_string(), kind: CNNLayerKind::MaxPool2D },
        ],
    };

    let flag = AbortFlag::new();

    let plan1 = CNNExecutionAdapter::build_plan(&graph, &flag).unwrap();
    let plan2 = CNNExecutionAdapter::build_plan(&graph, &flag).unwrap();

    assert_eq!(plan1.steps.len(), plan2.steps.len());
    for (s1, s2) in plan1.steps.iter().zip(plan2.steps.iter()) {
        assert_eq!(s1.kind, s2.kind);
        assert_eq!(s1.description, s2.description);
    }
}
