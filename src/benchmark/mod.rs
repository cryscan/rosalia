use std::error::Error;

use half::f16;
use vulkanalia::prelude::v1_4::*;

use crate::{
    app::{App, CommandBuffer, Tensor, Uniform},
    layout::{IndexFn, IntoLayout, Layout},
    num::{DataType, FromI8, Scalar},
};

mod coop;

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
