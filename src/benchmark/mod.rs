use std::error::Error;

use half::f16;
use vulkanalia::prelude::v1_4::*;

use crate::{
    app::{App, CommandBuffer, Tensor, Uniform},
    layout::{IndexFn, IntoLayout, Layout},
    num::{DataType, FromI8, Scalar},
};

mod coop;
mod ln;

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
    /// * `actual` - The actual results to check against the expected results.
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

#[derive(Debug)]
pub struct GemmBench<A, B, C, O> {
    pub app: App,

    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub layout_a: Layout,
    pub layout_b: Layout,
    pub layout_c: Layout,

    pub data_a: Vec<A>,
    pub data_b: Vec<B>,
    pub data_c: Vec<C>,

    pub tensor_a_host: Tensor<A>,
    pub tensor_b_host: Tensor<B>,
    pub tensor_c_host: Tensor<C>,
    pub tensor_o_host: Tensor<O>,

    pub tensor_a_device: Tensor<A>,
    pub tensor_b_device: Tensor<B>,
    pub tensor_c_device: Tensor<C>,
    pub tensor_o_device: Tensor<O>,

    pub params: Uniform,

    pub transfer: CommandBuffer,
    pub compute: CommandBuffer,
}

impl<A, B, C, O> GemmBench<A, B, C, O>
where
    A: Scalar + FromI8,
    B: Scalar + FromI8,
    C: Scalar + FromI8,
    O: Scalar,
{
    pub fn new(app: &App, m: usize, n: usize, k: usize) -> Result<Self, Box<dyn Error>> {
        let app = app.clone();

        let layout_a = Layout::from_shape([m, k]);
        let layout_b = Layout::from_shape([k, n]);
        let layout_c = Layout::from_shape([m, n]);

        let data_a = create_data(&layout_a);
        let data_b = create_data(&layout_b);
        let data_c = create_data(&layout_c);

        let tensor_a_host = app.create_tensor(&layout_a, None, true)?;
        let tensor_b_host = app.create_tensor(&layout_b, None, true)?;
        let tensor_c_host = app.create_tensor(&layout_c, None, true)?;
        let tensor_o_host = app.create_tensor(&layout_c, None, true)?;

        tensor_a_host.copy_from(&data_a, 0)?;
        tensor_b_host.copy_from(&data_b, 0)?;
        tensor_c_host.copy_from(&data_c, 0)?;
        tensor_o_host.clear()?;

        let tensor_a_device = app.create_tensor(&layout_a, None, false)?;
        let tensor_b_device = app.create_tensor(&layout_b, None, false)?;
        let tensor_c_device = app.create_tensor(&layout_c, None, false)?;
        let tensor_o_device = app.create_tensor(&layout_c, None, false)?;

        let params = [
            tensor_a_device.address,
            tensor_b_device.address,
            tensor_c_device.address,
            tensor_o_device.address,
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
            tensor_a_device.cmd_copy_from(transfer[0], &tensor_a_host);
            tensor_b_device.cmd_copy_from(transfer[0], &tensor_b_host);
            tensor_c_device.cmd_copy_from(transfer[0], &tensor_c_host);
            app.device.end_command_buffer(transfer[0])?;

            // transfer output initialization data to device
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[1], &info)?;
            tensor_o_device.cmd_copy_from(transfer[1], &tensor_o_host);
            app.device.end_command_buffer(transfer[1])?;

            // transfer output from device to host
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[2], &info)?;
            tensor_o_host.cmd_copy_from(transfer[2], &tensor_o_device);
            app.device.end_command_buffer(transfer[2])?;
        }

        Ok(Self {
            app,
            m,
            n,
            k,
            layout_a,
            layout_b,
            layout_c,
            data_a,
            data_b,
            data_c,
            tensor_a_host,
            tensor_b_host,
            tensor_c_host,
            tensor_o_host,
            tensor_a_device,
            tensor_b_device,
            tensor_c_device,
            tensor_o_device,
            params,
            transfer,
            compute,
        })
    }
}

impl<A, B, C, O> GemmBench<A, B, C, O>
where
    A: Scalar,
    B: Scalar,
    C: Scalar,
    O: Scalar,
{
    pub fn matches_type(&self, a: DataType, b: DataType, c: DataType, o: DataType) -> bool {
        (a, b, c, o) == (A::DATA_TYPE, B::DATA_TYPE, C::DATA_TYPE, O::DATA_TYPE)
    }
}

impl Bench for GemmBench<f16, f16, f16, f16> {
    type Output = f16;

    fn compute_ans(&self) -> Vec<Self::Output> {
        let Self {
            data_a,
            data_b,
            data_c,
            layout_a,
            layout_b,
            layout_c,
            ..
        } = self;
        let (m, n, k) = (self.m, self.n, self.k);

        let mut ans = vec![f16::ZERO; layout_c.co_size()];
        for (i, j) in itertools::iproduct!(0..m, 0..n) {
            let mut dot = data_c[layout_c.value([i, j])];
            for k in 0..k {
                let a = data_a[layout_a.value([i, k])];
                let b = data_b[layout_b.value([k, j])];
                dot += a * b;
            }
            ans[layout_c.value([i, j])] = dot;
        }
        ans
    }

    fn check_ans(&self, actual: &[Self::Output]) -> usize {
        self.layout_c
            .iter_indices()
            .map(|(_, value)| (actual[value].to_f32(), self.tensor_o_host[value].to_f32()))
            .filter(|(x, y)| (x - y).abs() > f16::EPSILON.to_f32())
            .count()
    }
}

impl Bench for GemmBench<f16, f16, f32, f32> {
    type Output = f32;

    fn compute_ans(&self) -> Vec<Self::Output> {
        let Self {
            data_a,
            data_b,
            data_c,
            layout_a,
            layout_b,
            layout_c,
            ..
        } = self;
        let (m, n, k) = (self.m, self.n, self.k);

        let mut ans = vec![0.0f32; layout_c.co_size()];
        for (i, j) in itertools::iproduct!(0..m, 0..n) {
            let mut dot = data_c[layout_c.value([i, j])];
            for k in 0..k {
                let a = data_a[layout_a.value([i, k])].to_f32();
                let b = data_b[layout_b.value([k, j])].to_f32();
                dot += a * b;
            }
            ans[layout_c.value([i, j])] = dot;
        }
        ans
    }

    fn check_ans(&self, actual: &[Self::Output]) -> usize {
        self.layout_c
            .iter_indices()
            .map(|(_, value)| (actual[value], self.tensor_o_host[value]))
            .filter(|(x, y)| (x - y).abs() > f16::EPSILON.to_f32())
            .count()
    }
}

/// Benchmark for LayerNorm operation.
///
/// Validates the correctness of the norm.comp shader for layer normalization.
/// The shader computes: output = ((x - mean) / sqrt(variance + epsilon)) * gamma + beta.
#[derive(Debug)]
pub struct LayerNormBench<I, O> {
    pub app: App,

    pub channel: usize,
    pub token: usize,
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
        token: usize,
        batch: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let app = app.clone();

        let layout_input = Layout::from_shape([channel, token, batch]);
        let layout_gamma = Layout::from_shape([channel]);
        let layout_beta = Layout::from_shape([channel]);

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
            token,
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
            token,
            batch,
            ..
        } = self;

        let epsilon: f32 = 1.0e-5;
        let mut ans = vec![f16::ZERO; layout_input.co_size()];

        for batch in 0..*batch {
            for token in 0..*token {
                // compute mean and variance for this token
                let mut sum: f32 = 0.0;
                let mut sq_sum: f32 = 0.0;

                for channel in 0..*channel {
                    let idx = layout_input.value([channel, token, batch]);
                    let value = data_input[idx].to_f32();
                    sum += value;
                    sq_sum += value * value;
                }

                let mean = sum / (*channel as f32);
                let variance = sq_sum / (*channel as f32) - mean * mean;
                let inv_std = 1.0 / (variance + epsilon).sqrt();

                // normalize and apply affine transformation
                for channel in 0..*channel {
                    let idx = layout_input.value([channel, token, batch]);
                    let value = data_input[idx].to_f32();
                    let normalized = (value - mean) * inv_std;

                    let g = data_gamma[layout_gamma.value([channel])].to_f32();
                    let b_val = data_beta[layout_beta.value([channel])].to_f32();
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
