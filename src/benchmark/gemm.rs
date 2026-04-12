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

const TILE_M: [usize; 2] = [128, 256];
const TILE_N: [usize; 2] = [128, 256];
const TILE_K: [usize; 2] = [32, 64];

// divide tiles into 4x2 subtiles
const NUM_SUBTILE_TILE_M: usize = 4;
const NUM_SUBTILE_TILE_N: usize = 2;

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
                let a = data_a[layout_a.value([k, i])];
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
                let a = data_a[layout_a.value([k, i])].to_f32();
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

impl<A, B, C, O> GemmBench<A, B, C, O>
where
    A: Scalar,
    B: Scalar,
    C: Scalar,
    O: Scalar,
    GemmBench<A, B, C, O>: Bench<Output = O>,
{
    pub fn benchmark_cooperative_matrix(&self) -> Result<(), Box<dyn Error>> {
        let Self {
            app,
            layout_a,
            layout_b,
            layout_c,
            tensor_o_host,
            params,
            transfer,
            compute,
            ..
        } = self;
        let (m, n, k) = (self.m, self.n, self.k);

        #[cfg(feature = "correctness")]
        let ans = self.compute_ans();

        unsafe {
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[0..=0]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;
        }

        for (tile, mat) in itertools::iproduct!(
            itertools::iproduct!(TILE_M, TILE_N, TILE_K),
            app.properties.cooperative_matrix.iter()
        ) {
            let max_sh_size = app.properties.limits.max_compute_shared_memory_size as usize;
            let sh_a_size = tile.0 * tile.2 * A::DATA_TYPE.size();
            let sh_b_size = tile.1 * tile.2 * B::DATA_TYPE.size();

            if !self.matches_type(mat.a, mat.b, mat.c, mat.o) {
                continue;
            }
            if sh_a_size + sh_b_size >= max_sh_size {
                continue;
            }

            let subtile = (tile.0 / NUM_SUBTILE_TILE_M, tile.1 / NUM_SUBTILE_TILE_N);

            log::info!("feature: cooperative matrix");
            log::info!(
                "tile: {:>3} x {:>3} x {:>3}, subtile: {:>2} x {:>3}, mat: {:>2} x {:>2} x {:>2}, {:>3} x {:>3} + {:>3} → {:>3}",
                tile.0,
                tile.1,
                tile.2,
                subtile.0,
                subtile.1,
                mat.m,
                mat.n,
                mat.k,
                A::DATA_TYPE,
                B::DATA_TYPE,
                C::DATA_TYPE,
                O::DATA_TYPE,
            );

            let layout_a_tile = layout_a.div_tiler([tile.2, tile.0])?;
            let layout_b_tile = layout_b.div_tiler([tile.2, tile.1])?;
            let layout_c_tile = layout_c
                .div_tiler([tile.0, tile.1])?
                .div_tiler([subtile.0, subtile.1])?
                .div_tiler([mat.m, mat.n])?;

            let layout_sh_a_tile = Layout::from_shape([tile.2, tile.0]);
            let layout_sh_b_tile = Layout::from_shape([tile.2, tile.1]);

            let layout_sh_a_subtile = layout_sh_a_tile
                .div_tiler([tile.2, subtile.0])?
                .div_tiler([mat.k, mat.m])?;
            let layout_sh_b_subtile = layout_sh_b_tile
                .div_tiler([tile.2, subtile.1])?
                .div_tiler([mat.k, mat.n])?;

            let pad = 16usize / A::DATA_TYPE.size();
            let layout_sh_a_tile = layout_sh_a_tile.pad_2d(pad);
            let layout_sh_b_tile = layout_sh_b_tile.pad_2d(pad);
            let layout_sh_a_subtile = layout_sh_a_subtile.pad_2d(pad);
            let layout_sh_b_subtile = layout_sh_b_subtile.pad_2d(pad);

            log::info!("\ttiled a: {layout_a_tile}");
            log::info!("\ttiled b: {layout_b_tile}");
            log::info!("\ttiled c: {layout_c_tile}");
            log::info!("\tshared a: {layout_sh_a_tile}");
            log::info!("\tshared b: {layout_sh_b_tile}");
            log::info!("\tshared a: {layout_sh_a_subtile}");
            log::info!("\tshared b: {layout_sh_b_subtile}");

            assert_eq!(layout_c_tile.len(), 8);
            assert_eq!(layout_sh_a_subtile.len(), 6);
            assert_eq!(layout_sh_b_subtile.len(), 6);

            tensor_o_host.clear()?;

            let specialization = [
                mat.m as u32,                            // MAT_M
                mat.n as u32,                            // MAT_N
                mat.k as u32,                            // MAT_K
                tile.0 as u32,                           // TILE_M
                tile.1 as u32,                           // TILE_N
                tile.2 as u32,                           // TILE_K
                m as u32,                                // M
                n as u32,                                // N
                k as u32,                                // K
                layout_a_tile.stride_of(0) as u32,       // STRIDE_A_TILE_X
                layout_a_tile.stride_of(1) as u32,       // STRIDE_A_TILE_Y
                layout_a_tile.stride_of(2) as u32,       // STRIDE_A_TILE_Z
                layout_a_tile.stride_of(3) as u32,       // STRIDE_A_TILE_W
                layout_b_tile.stride_of(0) as u32,       // STRIDE_B_TILE_X
                layout_b_tile.stride_of(1) as u32,       // STRIDE_B_TILE_Y
                layout_b_tile.stride_of(2) as u32,       // STRIDE_B_TILE_Z
                layout_b_tile.stride_of(3) as u32,       // STRIDE_B_TILE_W
                layout_sh_a_tile.stride_of(0) as u32,    // STRIDE_SH_A_TILE_X
                layout_sh_a_tile.stride_of(1) as u32,    // STRIDE_SH_A_TILE_Y
                layout_sh_b_tile.stride_of(0) as u32,    // STRIDE_SH_B_TILE_X
                layout_sh_b_tile.stride_of(1) as u32,    // STRIDE_SH_B_TILE_Y
                layout_sh_a_subtile.stride_of(0) as u32, // STRIDE_SH_A_MAT_X
                layout_sh_a_subtile.stride_of(1) as u32, // STRIDE_SH_A_MAT_Y
                layout_sh_a_subtile.stride_of(2) as u32, // STRIDE_SH_A_SUBTILE_X
                layout_sh_a_subtile.stride_of(3) as u32, // STRIDE_SH_A_SUBTILE_Y
                layout_sh_a_subtile.stride_of(5) as u32, // STRIDE_SH_A_SUBTILE_Z
                layout_sh_b_subtile.stride_of(0) as u32, // STRIDE_SH_B_MAT_X
                layout_sh_b_subtile.stride_of(1) as u32, // STRIDE_SH_B_MAT_Y
                layout_sh_b_subtile.stride_of(2) as u32, // STRIDE_SH_B_SUBTILE_X
                layout_sh_b_subtile.stride_of(3) as u32, // STRIDE_SH_B_SUBTILE_Y
                layout_sh_b_subtile.stride_of(5) as u32, // STRIDE_SH_B_SUBTILE_Z
                layout_c_tile.stride_of(0) as u32,       // STRIDE_C_SUBTILE_X
                layout_c_tile.stride_of(1) as u32,       // STRIDE_C_SUBTILE_Y
                layout_c_tile.stride_of(2) as u32,       // STRIDE_C_SUBTILE_Z
                layout_c_tile.stride_of(3) as u32,       // STRIDE_C_SUBTILE_W
                layout_c_tile.stride_of(4) as u32,       // STRIDE_C_TILE_X
                layout_c_tile.stride_of(5) as u32,       // STRIDE_C_TILE_Y
                layout_c_tile.stride_of(6) as u32,       // STRIDE_C_TILE_Z
                layout_c_tile.stride_of(7) as u32,       // STRIDE_C_TILE_W
                0u32,                                    // STRIDE_A_BATCH
                0u32,                                    // STRIDE_B_BATCH
                0u32,                                    // STRIDE_C_BATCH
            ];
            log::info!("\tspecialization: {:?}", specialization);

            let dt_a = A::DATA_TYPE.to_string().to_lowercase();
            let dt_c = C::DATA_TYPE.to_string().to_lowercase();
            let path = format!("shaders/spv/gemm_{dt_a}_{dt_c}_affine.spv");
            let file = Asset::get(&path).ok_or("failed to find shader")?;
            let shader = file.data;

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

                let info = vk::SubmitInfo::builder().command_buffers(&transfer[1..=1]);
                let queue = app.transfer.queue;
                app.device.queue_submit(queue, &[info], vk::Fence::null())?;
                app.device.queue_wait_idle(queue)?;

                let info = vk::CommandBufferBeginInfo::builder();
                app.device.begin_command_buffer(compute[0], &info)?;
                kernel.cmd_bind(compute[0], &[]);
                for _ in 0..REPEAT {
                    app.device.cmd_dispatch(
                        compute[0],
                        layout_c_tile.shape_of(6) as u32,
                        layout_c_tile.shape_of(7) as u32,
                        1,
                    );
                }
                app.device.end_command_buffer(compute[0])?;

                #[cfg(not(feature = "correctness"))]
                for _ in 0..WARMUP {
                    let info = vk::SubmitInfo::builder().command_buffers(compute);
                    let queue = app.compute.queue;
                    app.device.queue_submit(queue, &[info], vk::Fence::null())?;
                    app.device.queue_wait_idle(queue)?;
                }

                let info = vk::SubmitInfo::builder().command_buffers(compute);
                let queue = app.compute.queue;
                app.device.queue_submit(queue, &[info], vk::Fence::null())?;

                #[cfg(not(feature = "correctness"))]
                {
                    let timer = std::time::Instant::now();
                    app.device.queue_wait_idle(queue)?;
                    let duration = timer.elapsed();

                    let ops = 2 * m * n * k * REPEAT; // 2 operations per multiply-add
                    let duration = duration.as_secs_f64();
                    let tflops = (ops as f64) / (duration * 1e12);
                    log::info!("\tTFLOPS: {tflops:.3}");
                    log::info!("\tduration: {duration:?}, operations: {ops}, repeat: {REPEAT}");
                }
                #[cfg(feature = "correctness")]
                app.device.queue_wait_idle(queue)?;

                let info = vk::SubmitInfo::builder().command_buffers(&transfer[2..=2]);
                let queue = app.transfer.queue;
                app.device.queue_submit(queue, &[info], vk::Fence::null())?;
                app.device.queue_wait_idle(queue)?;
            }

            #[cfg(feature = "correctness")]
            assert_eq!(self.check_ans(&ans), 0);
        }

        Ok(())
    }
}
