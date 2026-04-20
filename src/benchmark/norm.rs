//! Benchmark for the `norm` compute shader.
//!
//! # Norm Operation
//!
//! The norm shader implements a unified normalization operator that supports both
//! Layer Normalization and Group Normalization through the same compute kernel,
//! differentiated only by the stride configuration of the parameter tensors.
//!
//! ## Dimensions
//!
//! - **Channel (C)**: The normalization dimension. Mean and variance are computed
//!   across all channels within each group independently.
//! - **Head (H)**: The number of groups. Each head computes its own normalization
//!   statistics (mean and inverse standard deviation) independently.
//! - **Batch**: Allows processing multiple independent instances in a single dispatch.
//!
//! ## Layer Norm vs Group Norm
//!
//! Both normalization variants use the same shader with different head counts
//! and stride configurations:
//!
//! ### Layer Norm (H = 1)
//!
//! In Layer Norm, the entire channel dimension is treated as a single group (H = 1).
//! Mean and variance are computed over all channels for each batch element independently.
//! Gamma and Beta are per-channel parameters shared across all batches.
//!
//! ### Group Norm (H > 1)
//!
//! In Group Norm, the channel dimension is divided into H groups. Each group computes
//! its own mean and variance independently, then applies the affine transformation.
//! Gamma and Beta remain per-channel parameters shared across batches, but with
//! group-dependent strides.
//!
//! Stride configuration for Gamma/Beta:
//! - `STRIDE_G_Y` = C, `STRIDE_G_Z` = 0 — Gamma is indexed as [channel, head],
//!   shared across batches.
//! - `STRIDE_B_Y` = C, `STRIDE_B_Z` = 0 — Beta is indexed as [channel, head],
//!   shared across batches.
//!
//! ### The `STRIDE_*_Z = 0` Technique
//!
//! Setting `STRIDE_G_Z = 0` and `STRIDE_B_Z = 0` is a deliberate technique that
//! makes Gamma and Beta invariant to the batch dimension. Since Gamma and Beta are
//! model-level parameters (fixed during inference), they should be shared across all
//! batch elements. The shader computes the element offset as:
//!
//! ```text
//! g_base = batch * STRIDE_G.z + head * STRIDE_G.y
//! ```
//!
//! By zeroing `STRIDE_G.z`, the `batch * STRIDE_G.z` term vanishes, causing all
//! batches to reference the same Gamma/Beta data. This avoids the need for a separate
//! "broadcast" code path in the shader.

use std::error::Error;

use half::f16;
use vulkanalia::prelude::v1_4::*;

use super::{Bench, create_data};
use crate::{
    app::{App, CommandBuffer, Tensor, Uniform},
    asset::Asset,
    layout::{IndexFn, Layout},
    num::{DataType, FromI8, Scalar},
};

/// Benchmark for the unified norm operator (Layer Norm / Group Norm).
///
/// When `head = 1`, this behaves as Layer Norm (normalization over the full channel
/// dimension per batch element). When `head > 1`, it behaves as Group Norm (channels
/// are divided into `head` groups, each normalized independently). See the module-level
/// documentation for details on stride configuration.
#[derive(Debug)]
pub struct LayerNormBench<I, O> {
    pub app: App,

    pub channel: usize,
    pub head: usize,
    pub batch: usize,

    pub layout_input: Layout,
    pub layout_gamma: Layout,
    pub layout_beta: Layout,

    pub data_input: Vec<I>,
    pub data_gamma: Vec<I>,
    pub data_beta: Vec<I>,

    pub tensor_input_host: Tensor<I>,
    pub tensor_gamma_host: Tensor<I>,
    pub tensor_beta_host: Tensor<I>,
    pub tensor_output_host: Tensor<O>,

    pub tensor_input_device: Tensor<I>,
    pub tensor_gamma_device: Tensor<I>,
    pub tensor_beta_device: Tensor<I>,
    pub tensor_output_device: Tensor<O>,

    pub params: Uniform,

    pub transfer: CommandBuffer,
    pub compute: CommandBuffer,
}

impl<I, O> LayerNormBench<I, O>
where
    I: Scalar + FromI8,
    O: Scalar + FromI8,
{
    pub fn new(
        app: &App,
        channel: usize,
        head: usize,
        batch: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let app = app.clone();

        let layout_input = Layout::from_shape([channel, head, batch]);
        let layout_gamma = Layout::from_shape([channel, head, 1]);
        let layout_beta = Layout::from_shape([channel, head, 1]);

        let data_input = create_data(&layout_input);
        let data_gamma = create_data(&layout_gamma);
        let data_beta = create_data(&layout_beta);

        // host tensors (mapped)
        let tensor_input_host = app.create_tensor(&layout_input, None, true)?;
        let tensor_gamma_host = app.create_tensor(&layout_gamma, None, true)?;
        let tensor_beta_host = app.create_tensor(&layout_beta, None, true)?;
        let tensor_output_host = app.create_tensor(&layout_input, None, true)?;

        tensor_input_host.copy_from(&data_input, 0)?;
        tensor_gamma_host.copy_from(&data_gamma, 0)?;
        tensor_beta_host.copy_from(&data_beta, 0)?;
        tensor_output_host.clear()?;

        // device tensors (not mapped)
        let tensor_input_device = app.create_tensor(&layout_input, None, false)?;
        let tensor_gamma_device = app.create_tensor(&layout_gamma, None, false)?;
        let tensor_beta_device = app.create_tensor(&layout_beta, None, false)?;
        let tensor_output_device = app.create_tensor(&layout_input, None, false)?;

        // uniform buffer with device addresses
        let params = [
            tensor_input_device.address,
            tensor_gamma_device.address,
            tensor_beta_device.address,
            tensor_output_device.address,
        ];
        let params = app
            .create_uniform(size_of_val(&params))?
            .copy_from(&params)?;

        let transfer = app.allocate_transfer_command_buffers(3)?;
        let compute = app.allocate_compute_command_buffers(1)?;

        unsafe {
            // transfer input tensors to device
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[0], &info)?;
            tensor_input_device.cmd_copy_from(transfer[0], &tensor_input_host);
            tensor_gamma_device.cmd_copy_from(transfer[0], &tensor_gamma_host);
            tensor_beta_device.cmd_copy_from(transfer[0], &tensor_beta_host);
            app.device.end_command_buffer(transfer[0])?;

            // transfer output initialization data to device
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[1], &info)?;
            tensor_output_device.cmd_copy_from(transfer[1], &tensor_output_host);
            app.device.end_command_buffer(transfer[1])?;

            // transfer output from device to host
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[2], &info)?;
            tensor_output_host.cmd_copy_from(transfer[2], &tensor_output_device);
            app.device.end_command_buffer(transfer[2])?;
        }

        Ok(Self {
            app,
            channel,
            head,
            batch,
            layout_input,
            layout_gamma,
            layout_beta,
            data_input,
            data_gamma,
            data_beta,
            tensor_input_host,
            tensor_gamma_host,
            tensor_beta_host,
            tensor_output_host,
            tensor_input_device,
            tensor_gamma_device,
            tensor_beta_device,
            tensor_output_device,
            params,
            transfer,
            compute,
        })
    }
}

impl<I, O> LayerNormBench<I, O>
where
    I: Scalar,
    O: Scalar,
{
    pub fn matches_type(&self, i: DataType, o: DataType) -> bool {
        (i, o) == (I::DATA_TYPE, O::DATA_TYPE)
    }
}

impl Bench for LayerNormBench<f16, f16> {
    type Output = f16;

    fn compute_ans(&self) -> Vec<Self::Output> {
        let Self {
            data_input,
            data_gamma,
            data_beta,
            layout_input,
            layout_gamma,
            layout_beta,
            channel,
            head,
            batch,
            ..
        } = self;

        let epsilon: f32 = 1.0e-5;
        let mut ans = vec![f16::ZERO; layout_input.co_size()];

        for batch in 0..*batch {
            for head in 0..*head {
                // compute mean and variance for this head
                let mut sum: f32 = 0.0;
                let mut sq_sum: f32 = 0.0;

                for channel in 0..*channel {
                    let idx = layout_input.value([channel, head, batch]);
                    let value = data_input[idx].to_f32();
                    sum += value;
                    sq_sum += value * value;
                }

                let mean = sum / (*channel as f32);
                let variance = sq_sum / (*channel as f32) - mean * mean;
                let inv_std = 1.0 / (variance + epsilon).sqrt();

                // normalize and apply affine transformation
                for channel in 0..*channel {
                    let idx = layout_input.value([channel, head, batch]);
                    let value = data_input[idx].to_f32();
                    let normalized = (value - mean) * inv_std;

                    let g = data_gamma[layout_gamma.value([channel, head, 0])].to_f32();
                    let b_val = data_beta[layout_beta.value([channel, head, 0])].to_f32();
                    let output = normalized * g + b_val;

                    ans[idx] = f16::from_f32(output);
                }
            }
        }

        ans
    }

    fn check_ans(&self, actual: &[Self::Output]) -> usize {
        self.layout_input
            .iter_indices()
            .map(|(_, value)| {
                (
                    actual[value].to_f32(),
                    self.tensor_output_host[value].to_f32(),
                )
            })
            .filter(|(x, y)| (x - y).abs() > 1e-2) // use slightly larger tolerance for f16
            .count()
    }
}

impl<I, O> LayerNormBench<I, O>
where
    I: Scalar,
    O: Scalar,
    LayerNormBench<I, O>: Bench<Output = O>,
{
    pub fn benchmark_layer_norm(&self) -> Result<(), Box<dyn Error>> {
        let Self {
            app,
            layout_input,
            layout_gamma,
            layout_beta,
            params,
            transfer,
            compute,
            channel,
            head,
            batch,
            ..
        } = self;

        #[cfg(feature = "correctness")]
        let ans = self.compute_ans();

        unsafe {
            // transfer input tensors to device
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[0..=0]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;
        }

        // check type match
        if !self.matches_type(DataType::F16, DataType::F16) {
            return Err("Type mismatch: only f16 -> f16 is supported".into());
        }

        // load shader
        let path = "shaders/spv/norm_f16_f16_affine.spv";
        let file = Asset::get(path).ok_or("failed to find shader")?;
        let shader = file.data;

        // specialization constants (matching norm.comp)
        let specialization = [
            *channel as u32,                  // C (hidden size)
            *head as u32,                     // H (number of heads)
            layout_input.stride_of(0) as u32, // STRIDE_X_X
            layout_input.stride_of(1) as u32, // STRIDE_X_Y
            layout_input.stride_of(2) as u32, // STRIDE_X_Z
            layout_input.stride_of(0) as u32, // STRIDE_Y_X
            layout_input.stride_of(1) as u32, // STRIDE_Y_Y
            layout_input.stride_of(2) as u32, // STRIDE_Y_Z
            layout_gamma.stride_of(0) as u32, // STRIDE_G_X
            layout_gamma.stride_of(1) as u32, // STRIDE_G_Y
            0u32,                             // STRIDE_G_Z
            layout_beta.stride_of(0) as u32,  // STRIDE_B_X
            layout_beta.stride_of(1) as u32,  // STRIDE_B_Y
            0u32,                             // STRIDE_B_Z
            1.0e-5f32.to_bits(),              // EPSILON (use default 1e-5)
        ];

        log::info!("feature: layer norm");
        log::info!("hidden_size: {channel}, heads: {head}, batch: {batch}");
        log::info!("\tspecialization: {:?}", specialization);

        let bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let kernel = app.create_kernel(&shader, &specialization, &bindings)?;
        kernel.binder().bind_uniform(params, 0, 0).build();

        unsafe {
            let flags = vk::CommandBufferResetFlags::empty();
            app.device.reset_command_buffer(compute[0], flags)?;

            // transfer output init data to device
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[1..=1]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;

            // record compute command buffer
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(compute[0], &info)?;
            kernel.cmd_bind(compute[0], &[]);
            // dispatch: one workgroup per (head, batch)
            app.device
                .cmd_dispatch(compute[0], *head as u32, *batch as u32, 1);
            app.device.end_command_buffer(compute[0])?;

            // execute compute
            let info = vk::SubmitInfo::builder().command_buffers(compute);
            let queue = app.compute.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;

            // transfer output back to host
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[2..=2]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;
        }

        #[cfg(feature = "correctness")]
        assert_eq!(self.check_ans(&ans), 0);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use half::f16;
    use simplelog::{LevelFilter, SimpleLogger};

    use super::*;
    use crate::app::App;

    fn init_test() {
        _ = SimpleLogger::init(LevelFilter::Debug, Default::default());
        fastrand::seed(514);
    }

    #[test]
    fn test_layer_norm_f16_f16() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        // small sizes for quick testing
        let c = 256;
        let h = 16;
        let b = 256;

        let bench = LayerNormBench::<f16, f16>::new(&app, c, h, b)?;
        bench.benchmark_layer_norm()
    }

    #[test]
    fn test_layer_norm_f16_f16_small() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        // minimal sizes for basic correctness
        let c = 8;
        let h = 2;
        let b = 1;

        let bench = LayerNormBench::<f16, f16>::new(&app, c, h, b)?;
        bench.benchmark_layer_norm()
    }
}
