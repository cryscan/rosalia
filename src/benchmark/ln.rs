use std::error::Error;

use vulkanalia::prelude::v1_4::*;

use super::{Bench, LayerNormBench};
use crate::{
    asset::Asset,
    num::{DataType, Scalar},
};

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
            params,
            transfer,
            compute,
            channel,
            token,
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
            *token as u32,                    // T (number of tokens)
            layout_input.stride_of(1) as u32, // STRIDE_INPUT_Y
            layout_input.stride_of(2) as u32, // STRIDE_INPUT_Z
            1.0e-5f32.to_bits(),              // EPSILON (use default 1e-5)
        ];

        log::info!("feature: layer norm");
        log::info!("hidden_size: {channel}, tokens: {token}, batch: {batch}");
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
            // dispatch: one workgroup per (token, batch)
            app.device
                .cmd_dispatch(compute[0], *token as u32, *batch as u32, 1);
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
        {
            let mismatches = self.check_ans(&ans);
            if mismatches > 0 {
                log::error!(
                    "LayerNorm correctness check failed: {} mismatches",
                    mismatches
                );
            }
            assert_eq!(mismatches, 0, "LayerNorm correctness check failed");
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
    fn test_layer_norm_f16_f16() {
        let app = init_test();

        // small sizes for quick testing
        let channel = 64;
        let token = 16;
        let batch = 4;

        let bench = LayerNormBench::<f16, f16>::new(&app, channel, token, batch)
            .expect("failed to create LayerNormBench");

        bench.benchmark_layer_norm().expect("benchmark failed");
    }

    #[test]
    fn test_layer_norm_f16_f16_small() {
        let app = init_test();

        // minimal sizes for basic correctness
        let c = 8;
        let t = 2;
        let batch = 1;

        let bench = LayerNormBench::<f16, f16>::new(&app, c, t, batch)
            .expect("failed to create LayerNormBench");

        bench.benchmark_layer_norm().expect("benchmark failed");
    }
}
