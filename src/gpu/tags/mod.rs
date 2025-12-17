use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct KernelTags {
    tags: HashSet<&'static str>,
}

impl KernelTags {
    pub fn new() -> Self {
        Self { tags: HashSet::new() }
    }

    pub fn with(mut self, tag: &'static str) -> Self {
        self.tags.insert(tag);
        self
    }

    pub fn contains(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    pub fn all(&self) -> Vec<&'static str> {
        self.tags.iter().cloned().collect()
    }
}
