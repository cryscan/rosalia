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

#[cfg(feature = "correctness")]
const REPEAT: usize = 1;
#[cfg(not(feature = "correctness"))]
const REPEAT: usize = 10;
#[cfg(not(feature = "correctness"))]
const WARMUP: usize = 5;

/// Benchmark for GEMV (matrix-vector multiplication) operation.
///
/// Computes: y = A * x, where A is (K, M) layout, x is (K), y is (M).
/// The A matrix layout matches gemm.comp's A matrix: small axis first (K, M).
#[derive(Debug)]
pub struct GemvBench<A, O> {
    pub app: App,

    pub m: usize,
    pub k: usize,
    pub batch: usize,

    pub layout_a: Layout,
    pub layout_x: Layout,
    pub layout_y: Layout,

    pub data_a: Vec<A>,
    pub data_x: Vec<A>,

    pub tensor_a_host: Tensor<A>,
    pub tensor_x_host: Tensor<A>,
    pub tensor_y_host: Tensor<O>,

    pub tensor_a_device: Tensor<A>,
    pub tensor_x_device: Tensor<A>,
    pub tensor_y_device: Tensor<O>,

    pub params: Uniform,

    pub transfer: CommandBuffer,
    pub compute: CommandBuffer,
}

impl<A, O> GemvBench<A, O>
where
    A: Scalar + FromI8,
    O: Scalar + FromI8,
{
    pub fn new(app: &App, m: usize, k: usize, batch: usize) -> Result<Self, Box<dyn Error>> {
        let app = app.clone();

        // A has layout (K, M) — small axis first, same as gemm.comp's A
        let layout_a = Layout::from_shape([k, m, batch]);
        let layout_x = Layout::from_shape([k, batch]);
        let layout_y = Layout::from_shape([m, batch]);

        let data_a = create_data(&layout_a);
        let data_x = create_data(&layout_x);

        // host tensors (mapped)
        let tensor_a_host = app.create_tensor(&layout_a, None, true)?;
        let tensor_x_host = app.create_tensor(&layout_x, None, true)?;
        let tensor_y_host = app.create_tensor(&layout_y, None, true)?;

        tensor_a_host.copy_from(&data_a, 0)?;
        tensor_x_host.copy_from(&data_x, 0)?;
        tensor_y_host.clear()?;

        // device tensors (not mapped)
        let tensor_a_device = app.create_tensor(&layout_a, None, false)?;
        let tensor_x_device = app.create_tensor(&layout_x, None, false)?;
        let tensor_y_device = app.create_tensor(&layout_y, None, false)?;

        // uniform buffer with device addresses
        let params = [
            tensor_a_device.address,
            tensor_x_device.address,
            tensor_y_device.address,
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
            tensor_x_device.cmd_copy_from(transfer[0], &tensor_x_host);
            app.device.end_command_buffer(transfer[0])?;

            // transfer output initialization data to device
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[1], &info)?;
            tensor_y_device.cmd_copy_from(transfer[1], &tensor_y_host);
            app.device.end_command_buffer(transfer[1])?;

            // transfer output from device to host
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[2], &info)?;
            tensor_y_host.cmd_copy_from(transfer[2], &tensor_y_device);
            app.device.end_command_buffer(transfer[2])?;
        }

        Ok(Self {
            app,
            m,
            k,
            batch,
            layout_a,
            layout_x,
            layout_y,
            data_a,
            data_x,
            tensor_a_host,
            tensor_x_host,
            tensor_y_host,
            tensor_a_device,
            tensor_x_device,
            tensor_y_device,
            params,
            transfer,
            compute,
        })
    }
}

impl<A, O> GemvBench<A, O>
where
    A: Scalar,
    O: Scalar,
{
    pub fn matches_type(&self, a: DataType, o: DataType) -> bool {
        (a, o) == (A::DATA_TYPE, O::DATA_TYPE)
    }
}

impl Bench for GemvBench<f16, f16> {
    type Output = f16;

    fn compute_ans(&self) -> Vec<Self::Output> {
        let Self {
            data_a,
            data_x,
            layout_a,
            layout_x,
            layout_y,
            m,
            k,
            batch,
            ..
        } = self;

        let mut ans = vec![f16::ZERO; layout_y.co_size()];
        for batch in 0..*batch {
            for row in 0..*m {
                let mut dot: f32 = 0.0;
                for col in 0..*k {
                    // A layout: (K, M, batch) — small axis first
                    let a_idx = layout_a.value([col, row, batch]);
                    let x_idx = layout_x.value([col, batch]);
                    let a_val = data_a[a_idx].to_f32();
                    let x_val = data_x[x_idx].to_f32();
                    dot += a_val * x_val;
                }
                let y_idx = layout_y.value([row, batch]);
                ans[y_idx] = f16::from_f32(dot);
            }
        }
        ans
    }

    fn check_ans(&self, actual: &[Self::Output]) -> usize {
        self.layout_y
            .iter_indices()
            .map(|(_, value)| (actual[value].to_f32(), self.tensor_y_host[value].to_f32()))
            .filter(|(x, y)| (x - y).abs() > 1e-2) // tolerance for f16 accumulation
            .count()
    }
}

impl<A, O> GemvBench<A, O>
where
    A: Scalar,
    O: Scalar,
    GemvBench<A, O>: Bench<Output = O>,
{
    pub fn benchmark_gemv(&self) -> Result<(), Box<dyn Error>> {
        let Self {
            app,
            layout_a,
            layout_x,
            layout_y,
            params,
            transfer,
            compute,
            m,
            k,
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
        let dt_a = A::DATA_TYPE.to_string().to_lowercase();
        let dt_o = O::DATA_TYPE.to_string().to_lowercase();
        let path = format!("shaders/spv/gemv_{dt_a}_{dt_o}.spv");
        let file = Asset::get(&path).ok_or("failed to find shader")?;
        let shader = file.data;

        // specialization constants (matching gemv.comp)
        let specialization = [
            *m as u32,                    // M (number of output elements)
            *k as u32,                    // K (contraction dimension)
            layout_a.stride_of(0) as u32, // STRIDE_A_X (always 1 for contiguous)
            layout_a.stride_of(1) as u32, // STRIDE_A_Y
            layout_a.stride_of(2) as u32, // STRIDE_A_Z (batch stride)
            layout_x.stride_of(0) as u32, // STRIDE_X_X (always 1 for contiguous)
            layout_x.stride_of(1) as u32, // STRIDE_X_Y (batch stride)
            layout_y.stride_of(0) as u32, // STRIDE_Y_X (always 1 for contiguous)
            layout_y.stride_of(1) as u32, // STRIDE_Y_Y (batch stride)
        ];

        log::info!("m: {m}, k: {k}, batch: {batch}");
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
            for _ in 0..REPEAT {
                // dispatch: one workgroup per (row, batch)
                app.device
                    .cmd_dispatch(compute[0], *m as u32, *batch as u32, 1);
            }
            app.device.end_command_buffer(compute[0])?;

            #[cfg(not(feature = "correctness"))]
            for _ in 0..WARMUP {
                let info = vk::SubmitInfo::builder().command_buffers(compute);
                let queue = app.compute.queue;
                app.device.queue_submit(queue, &[info], vk::Fence::null())?;
                app.device.queue_wait_idle(queue)?;
            }

            // execute compute
            let info = vk::SubmitInfo::builder().command_buffers(compute);
            let queue = app.compute.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;

            #[cfg(not(feature = "correctness"))]
            {
                let timer = std::time::Instant::now();
                app.device.queue_wait_idle(queue)?;
                let duration = timer.elapsed();

                let ops = 2 * m * k * batch * REPEAT;
                let duration = duration.as_secs_f64();
                let gflops = (ops as f64) / (duration * 1e9);
                log::info!("\tGFLOPS: {gflops:.3}");
                log::info!("\tduration: {duration:?}, operations: {ops}, repeat: {REPEAT}");
            }
            #[cfg(feature = "correctness")]
            app.device.queue_wait_idle(queue)?;

            // transfer output back to host
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[2..=2]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;
        }

        #[cfg(feature = "correctness")]
        {
            let mismatches = self.check_ans(&ans);
            if mismatches > 0 {
                log::error!("GEMV correctness check failed: {} mismatches", mismatches);
            }
            assert_eq!(mismatches, 0, "GEMV correctness check failed");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use half::f16;

    fn init_test() -> App {
        let _ = simplelog::SimpleLogger::init(
            simplelog::LevelFilter::Debug,
            simplelog::Config::default(),
        );
        fastrand::seed(514);
        App::new().expect("failed to create app")
    }

    #[test]
    fn test_gemv_f16_f16() {
        let app = init_test();

        // small sizes for quick testing
        let m = 64;
        let k = 128;
        let batch = 4;

        let bench =
            GemvBench::<f16, f16>::new(&app, m, k, batch).expect("failed to create GemvBench");

        bench.benchmark_gemv().expect("benchmark failed");
    }

    #[test]
    fn test_gemv_f16_f16_small() {
        let app = init_test();

        // minimal sizes for basic correctness
        let m = 8;
        let k = 16;
        let batch = 1;

        let bench =
            GemvBench::<f16, f16>::new(&app, m, k, batch).expect("failed to create GemvBench");

        bench.benchmark_gemv().expect("benchmark failed");
    }

    #[test]
    fn test_gemv_f16_f16_batch() {
        let app = init_test();

        // test with multiple batches
        let m = 32;
        let k = 64;
        let batch = 8;

        let bench =
            GemvBench::<f16, f16>::new(&app, m, k, batch).expect("failed to create GemvBench");

        bench.benchmark_gemv().expect("benchmark failed");
    }
}
