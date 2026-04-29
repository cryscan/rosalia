use crate::{layout::IntoLayout, num::FromI8};

mod gemm;
mod gemv;
mod norm;
mod token_shift;

pub use gemm::GemmBench;
pub use gemv::GemvBench;
pub use norm::LayerNormBench;
pub use token_shift::TokenShiftBench;

pub trait Bench {
    /// The output type of the computation.
    type Output;

    /// Computes the expected answer for the benchmark.
    ///
    /// # Returns
    /// A vector containing the expected results of the computation.
    fn compute_ans(&self) -> Vec<Self::Output>;

    /// Checks if the actual results match the expected results.
    ///
    /// # Arguments
    /// - `actual`: The actual results to check against the expected results.
    ///
    /// # Returns
    /// The number of mismatches between actual and expected results.
    fn check_ans(&self, actual: &[Self::Output]) -> usize;
}

fn create_data<T: FromI8>(layout: impl IntoLayout) -> Vec<T> {
    let layout = layout.into_layout();
    (0..layout.size())
        .map(|_| fastrand::i8(-2..=2))
        .map(T::from_i8)
        .collect()
}
