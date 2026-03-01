use std::error::Error;

use vulkanalia::prelude::v1_4::*;

use super::{Bench, GemmBench};
use crate::{asset::Asset, layout::Layout, num::Scalar};

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

            let layout_a_tile = layout_a.div_tiler([tile.0, tile.2])?;
            let layout_b_tile = layout_b.div_tiler([tile.2, tile.1])?;
            let layout_c_tile = layout_c
                .div_tiler([tile.0, tile.1])?
                .div_tiler([subtile.0, subtile.1])?
                .div_tiler([mat.m, mat.n])?;

            let layout_sh_a = Layout::from_shape([tile.0, tile.2]);
            let layout_sh_b = Layout::from_shape([tile.2, tile.1]);

            let layout_sh_a_subtile = layout_sh_a
                .div_tiler([subtile.0, tile.2])?
                .div_tiler([mat.m, mat.k])?;
            let layout_sh_b_subtile = layout_sh_b
                .div_tiler([tile.2, subtile.1])?
                .div_tiler([mat.k, mat.n])?;

            let pad = 16usize / A::DATA_TYPE.size();
            let layout_sh_a = layout_sh_a.pad_2d(pad);
            let layout_sh_b = layout_sh_b.pad_2d(pad);
            let layout_sh_a_subtile = layout_sh_a_subtile.pad_2d(pad);
            let layout_sh_b_subtile = layout_sh_b_subtile.pad_2d(pad);

            log::info!("\ttiled a: {layout_a_tile}");
            log::info!("\ttiled b: {layout_b_tile}");
            log::info!("\ttiled c: {layout_c_tile}");
            log::info!("\tshared a: {layout_sh_a}");
            log::info!("\tshared b: {layout_sh_b}");
            log::info!("\tshared a: {layout_sh_a_subtile}");
            log::info!("\tshared b: {layout_sh_b_subtile}");

            assert_eq!(layout_c_tile.len(), 8);
            assert_eq!(layout_sh_a_subtile.len(), 6);
            assert_eq!(layout_sh_b_subtile.len(), 6);

            tensor_o_host.clear()?;

            let specialization = [
                mat.m as u32,
                mat.n as u32,
                mat.k as u32,
                tile.0 as u32,
                tile.1 as u32,
                tile.2 as u32,
                m as u32,
                n as u32,
                k as u32,
                layout_a_tile.stride_of(1) as u32,
                layout_a_tile.stride_of(2) as u32,
                layout_a_tile.stride_of(3) as u32,
                layout_b_tile.stride_of(1) as u32,
                layout_b_tile.stride_of(2) as u32,
                layout_b_tile.stride_of(3) as u32,
                layout_sh_a.stride_of(1) as u32,
                layout_sh_b.stride_of(1) as u32,
                layout_sh_a_subtile.stride_of(2) as u32,
                layout_sh_a_subtile.stride_of(3) as u32,
                layout_sh_a_subtile.stride_of(5) as u32,
                layout_sh_b_subtile.stride_of(2) as u32,
                layout_sh_b_subtile.stride_of(3) as u32,
                layout_sh_b_subtile.stride_of(5) as u32,
                layout_sh_a_subtile.stride_of(1) as u32,
                layout_sh_b_subtile.stride_of(1) as u32,
                layout_c_tile.stride_of(4) as u32,
                layout_c_tile.stride_of(5) as u32,
                layout_c_tile.stride_of(6) as u32,
                layout_c_tile.stride_of(7) as u32,
                layout_c_tile.stride_of(1) as u32,
                layout_c_tile.stride_of(2) as u32,
                layout_c_tile.stride_of(3) as u32,
                0u32,
                0u32,
                0u32,
            ];
            println!("specialization: {:?}", specialization);

            let dt_a = A::DATA_TYPE.to_string().to_lowercase();
            let dt_c = C::DATA_TYPE.to_string().to_lowercase();
            let path = format!("shaders/spv/gemm_{dt_a}_{dt_c}.spv");
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
