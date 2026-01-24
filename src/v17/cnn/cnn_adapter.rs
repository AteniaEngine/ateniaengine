use crate::v17::cnn::conv2d::AbortFlag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CNNAdapterError {
    InvalidGraph(String),
    Aborted,
}

/// Minimal explicit CNN graph description used by the adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct CNNLayer {
    pub name: String,
    pub kind: CNNLayerKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CNNLayerKind {
    Conv2D,
    Bias,
    ReLU,
    MaxPool2D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNGraph {
    pub layers: Vec<CNNLayer>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CNNPlanStepKind {
    Conv2D,
    Bias,
    ReLU,
    MaxPool2D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNPlanStep {
    pub kind: CNNPlanStepKind,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CNNExecutionPlan {
    pub steps: Vec<CNNPlanStep>,
}

pub struct CNNExecutionAdapter;

impl CNNExecutionAdapter {
    pub fn build_plan(
        graph: &CNNGraph,
        abort_flag: &AbortFlag,
    ) -> Result<CNNExecutionPlan, CNNAdapterError> {
        if abort_flag.is_aborted() {
            return Err(CNNAdapterError::Aborted);
        }

        if graph.layers.is_empty() {
            return Err(CNNAdapterError::InvalidGraph("empty CNN graph".to_string()));
        }

        // Validate that the sequence only contains supported layer kinds
        // and that there is at most one Conv2D in this minimal adapter.
        let mut seen_conv = false;
        let mut seen_pool = false;

        for layer in &graph.layers {
            if abort_flag.is_aborted() {
                return Err(CNNAdapterError::Aborted);
            }

            match layer.kind {
                CNNLayerKind::Conv2D => {
                    if seen_conv {
                        return Err(CNNAdapterError::InvalidGraph(
                            "multiple Conv2D layers not supported in minimal adapter".to_string(),
                        ));
                    }
                    seen_conv = true;
                }
                CNNLayerKind::Bias => {
                    if !seen_conv {
                        return Err(CNNAdapterError::InvalidGraph(
                            "Bias must follow Conv2D".to_string(),
                        ));
                    }
                    if seen_pool {
                        return Err(CNNAdapterError::InvalidGraph(
                            "Bias cannot appear after MaxPool2D".to_string(),
                        ));
                    }
                }
                CNNLayerKind::ReLU => {
                    if !seen_conv {
                        return Err(CNNAdapterError::InvalidGraph(
                            "ReLU must follow Conv2D".to_string(),
                        ));
                    }
                    if seen_pool {
                        return Err(CNNAdapterError::InvalidGraph(
                            "ReLU cannot appear after MaxPool2D".to_string(),
                        ));
                    }
                }
                CNNLayerKind::MaxPool2D => {
                    if !seen_conv {
                        return Err(CNNAdapterError::InvalidGraph(
                            "MaxPool2D must follow Conv2D".to_string(),
                        ));
                    }
                    seen_pool = true;
                }
            }
        }

        // Map each CNN layer 1:1 to a plan step, without reordering or fusion.
        let mut steps: Vec<CNNPlanStep> = Vec::new();
        for layer in graph.layers.iter() {
            if abort_flag.is_aborted() {
                return Err(CNNAdapterError::Aborted);
            }

            let kind = match layer.kind {
                CNNLayerKind::Conv2D => CNNPlanStepKind::Conv2D,
                CNNLayerKind::Bias => CNNPlanStepKind::Bias,
                CNNLayerKind::ReLU => CNNPlanStepKind::ReLU,
                CNNLayerKind::MaxPool2D => CNNPlanStepKind::MaxPool2D,
            };

            steps.push(CNNPlanStep {
                kind,
                description: layer.name.clone(),
            });
        }

        Ok(CNNExecutionPlan { steps })
    }
}
